"""One-shot natural-language compilation trials; no gold-informed repair."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from thinking_runtime.constraints import IR_SCHEMA, LANGUAGE, validate, solve_constraints, verify_assignment, semantic_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',required=True)
    p.add_argument('--gguf',required=True)
    p.add_argument('--port',type=int,default=8089)
    p.add_argument('--seconds',type=float,default=120)
    args=p.parse_args()
    cases=json.loads((ROOT/'examples/compiler-cases.json').read_text())
    freeze=json.loads((ROOT/'reports/compiler-freeze.json').read_text())
    for path, digest in freeze['sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen source or task changed: '+path)
    report={'configuration':vars(args),'freeze':freeze,'rows':[],
            'scope':'Three newly authored instances in a limited constraint language, not unseen-in-training evidence.',
            'gold_used_for_inference':False,'one_model_call_per_case':True,'cost_usd':None,
            'notes':['No syntax or semantic repair using gold references.','References only audit candidate semantics and final answers.',
                     'Execution uses only model-generated IR.','No 120B baseline.','Laya, TimesFM and embeddings not used.']}
    output=ROOT/'reports/compiler-run.json'
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    command=[args.server,'-m',args.gguf,'-c','4096','-t','4','--parallel','1','--host','127.0.0.1','--port',str(args.port)]
    report['worker_command']=command
    (ROOT/'workspace').mkdir(exist_ok=True)
    with (ROOT/'workspace/compiler-server.log').open('w') as log:
        worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        load_start=time.perf_counter()
        try:
            while time.perf_counter()-load_start<120:
                if worker.poll() is not None: raise RuntimeError('Worker exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=1) as response:
                        if response.status==200: break
                except OSError: pass
                time.sleep(.25)
            else: raise TimeoutError('Worker startup timed out')
            report['cold_load_seconds']=time.perf_counter()-load_start
            for case in cases:
                row={'id':case['id'],'problem':case['problem']}
                report['rows'].append(row)
                payload={'model':'gemma-e2b-q4','messages':[{'role':'user','content':LANGUAGE+'\nProblem:\n'+case['problem']}],
                         'temperature':0,'seed':7,'max_tokens':1800,'cache_prompt':False,
                         'chat_template_kwargs':{'enable_thinking':False},
                         'response_format':{'type':'json_schema','json_schema':{'name':'constraint_ir','strict':True,'schema':IR_SCHEMA}}}
                row['request']=payload
                started=time.perf_counter()
                try:
                    request=urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=args.seconds) as response:
                        data=json.load(response)
                    row['compile_seconds']=time.perf_counter()-started
                    row['model_response']=data
                    ir=json.loads(data['choices'][0]['message']['content'])
                    validate(ir)
                    row['schema_valid']=True
                    row['compiled_ir']=ir
                    remaining=args.seconds-(time.perf_counter()-started)
                    if remaining<=0: raise TimeoutError('Compilation exhausted execution budget')
                    row['execution']=solve_constraints(ir,seconds=remaining)
                    row['pipeline_seconds']=time.perf_counter()-started
                    # This evaluation happens after execution and is not fed back.
                    audit_start=time.perf_counter()
                    row['semantic_audit']=semantic_audit(case['reference'],ir)
                    answer=row['execution']['assignment']
                    row['answer_satisfies_reference']=verify_assignment(case['reference'],answer) if answer is not None else None
                    row['end_to_end_pass']=bool(row['semantic_audit']['equivalent'] and
                        ((row['execution']['status']=='verified' and row['answer_satisfies_reference']) or
                         (row['execution']['status']=='unsatisfiable' and row['semantic_audit']['reference_solution_count']==0)))
                    row['audit_seconds']=time.perf_counter()-audit_start
                except Exception as exc:
                    row['error']=f'{type(exc).__name__}: {exc}'
                    row['end_to_end_pass']=False
                    row.setdefault('compile_seconds',time.perf_counter()-started)
                print(f"{case['id']}: end_to_end_pass={row['end_to_end_pass']}, compile={row['compile_seconds']:.3f}s",flush=True)
                save()
        except Exception as exc:
            report['worker_error']=f'{type(exc).__name__}: {exc}'
        finally:
            worker.terminate()
            try: worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill(); worker.wait()
            report['passed']=sum(row.get('end_to_end_pass',False) for row in report['rows'])
            report['planned_cases']=len(cases)
            save()
    print(f"Saved {output}; passed {report['passed']}/{len(cases)}",flush=True)
    if 'worker_error' in report: raise SystemExit(report['worker_error'])


if __name__=='__main__': main()
