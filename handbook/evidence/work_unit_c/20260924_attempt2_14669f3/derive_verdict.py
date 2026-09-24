"""Read-only derivation from C1 raw artifacts. Run from repository root."""
import json, math, hashlib, tarfile, sys
from pathlib import Path

sys.path.insert(0,str(Path.cwd()))
from fq_ssiuu.evaluation.identity import digest,file_digest
from fq_ssiuu.evaluation.certificate import load_report
from fq_ssiuu.evaluation.derived_summary import validate_utility_summary
from fq_ssiuu.evaluation.qra_pipeline import MetricObservation
from experiments.reproductions.QRA.metrics import text_metrics,truth_ratios,utility_components
from experiments.reproductions.QRA.validation_contract import expected_recipe
from experiments.reproductions.QRA.validate_runtime import evaluator_identity, expected_modules
from scripts.validate_og001_artifacts import validate_setup_artifacts

root=Path(__file__).resolve().parent
raw=root/'raw'
runs=[p for p in (raw/'runs').iterdir() if p.is_dir() and not p.name.endswith('-sealed')]
assert len(runs)==1
run=runs[0]
run_id=run.name
sealed=raw/'runs'/(run_id+'-sealed')
archive=json.loads((sealed/'ARCHIVE_MANIFEST.json').read_text())
assert file_digest(sealed/'artifacts.tar')==archive['archive']['sha256']
with tarfile.open(sealed/'artifacts.tar') as tar:
    assert set(tar.getnames())=={r['path'] for r in archive['artifacts']}
    for item in archive['artifacts']:
        contents=tar.extractfile(item['path']).read()
        assert hashlib.sha256(contents).hexdigest()==item['sha256']==file_digest(run/item['path'])
        assert len(contents)==item['size_bytes']
read=lambda name:json.loads((run/name).read_text())
provenance=read('source_provenance.json')
source=json.loads((root/'SOURCE_MANIFEST.json').read_text())
assert provenance['source_manifest']==source
assert provenance['source_origin_sha']=='14669f34cbaefde8c4e3d7d7f89616a240a6a6c7'
assert provenance['source_manifest_sha256']==file_digest(root/'SOURCE_MANIFEST.json')
staging=json.loads((root/'staging_receipt.json').read_text())
assert staging['staged_notebook_sha256']==file_digest(root/'submitted_notebook.ipynb')
assert read('recipe.json')==expected_recipe()
assert read('run_result.json')['status']=='PASS'
assert read('run_result.json')['scientific_experiment_executed'] is False
assert read('method_validation_execution.json')['returncode']==0
env=validate_setup_artifacts(run/'environment',provenance['snapshot_commit_sha'],Path('requirements/kaggle-native-baseline.json'))
assert env['status']=='PASS'
child=run/'method_validation'
result=read('method_validation/validation_result.json')
assert result['status']=='PASS' and result['run_id']==run_id
assert result['versions']==expected_recipe()['expected_versions']
assert result['optimizer_steps']==result['backward_calls']==0 and result['model_training'] is False
assert result['visible_devices']=='0' and result['device_count']==1 and 'T4' in result['gpu']
assert result['attack_detected'] is result['durability_pass'] is None
for memory in result['memory'].values():
    assert memory['peak_allocated_bytes']<16_000_000_000 and memory['peak_reserved_bytes']<16_000_000_000
    assert result['release_mechanism']=='sequential_process_exit'
assert len(result['worker_executions'])==4
assert all(e['process_reaped'] and e['returncode']==0 for e in result['worker_executions'])
assert all(a['finished_unix']<=b['started_unix'] for a,b in zip(result['worker_executions'],result['worker_executions'][1:]))
for item in read('method_validation/OUTPUT_MANIFEST.json')['artifacts']:
    assert file_digest(child/item['path'])==item['sha256']
report=load_report(child/'report.json')
assert report['schema_version']=='v5' and report['report_status']=='VALID_DESCRIPTIVE_RECORD'
assert report['run_id']==run_id
group=read('method_validation/comparison_group.json')
assert report['comparison_group']==group
assert group['evaluator_sha256']==evaluator_identity()
count=0
for precision in ('fp16','int8','nf4'):
    artifact=read('method_validation/'+precision+'_artifact.json')
    assert artifact==group['artifacts'][precision]['artifact_manifest']
    assert digest(artifact)==group['artifacts'][precision]['artifact_sha256']
    assert [m['name'] for m in artifact['quantized_modules']]==([] if precision=='fp16' else expected_modules({'model_type':'llama','num_hidden_layers':16}))
    rows=read('method_validation/'+precision+'_records.json')
    assert len(rows)==4
    for r in rows:
        assert digest(r['source_record'])==r['source_record_sha256']
        assert digest(r['tensors'])==r['tensors_sha256']
        assert digest(r['generation']['decoding_spec'])==r['generation']['decoding_sha256']
        assert math.isfinite(r['answer_log_probability'])
        computed=text_metrics(r['source_record']['answer'],r['generation']['text'])
        assert all(computed[k]==r[k] for k in ('rouge_l_recall','exact_match'))
        tr=truth_ratios(r['paraphrase_mean_nll'],r['perturbation_mean_nlls'])
        assert all(math.isclose(tr[k],r[k],rel_tol=1e-12) for k in ('tofu_truth_ratio','openunlearning_truth_ratio'))
        dataset={'retain_perturbed':'retain','real_authors_perturbed':'real_authors','world_facts_perturbed':'world_facts'}.get(r['source_split'])
        if dataset:
            assert utility_components(-r['answer_log_probability'],r['perturbation_mean_nlls'],r['rouge_l_recall'],dataset=dataset,paraphrase_mean_nll=r['paraphrase_mean_nll'])==r['utility_components']
        count+=1
observations=[MetricObservation(**x) for x in report['observations']]
for observation in observations:
    rows=read('method_validation/'+observation.precision+'_records.json')
    by_id={r['sample_id']:r for r in rows}
    assert all(r['measurement']==by_id[r['sample_id']] for r in observation.raw_records)
for summary in read('method_validation/utility_summaries.json'):
    validate_utility_summary(summary,[o for o in observations if o.precision==summary['precision'] and o.kind=='utility_score'],group)
verdict={'status':'PASS_WITHIN_MATCHED_PRECISION_INFERENCE_SCOPE','run_id':run_id,
         'origin_commit':provenance['source_origin_sha'],'source_tree_sha':provenance['source_tree_sha'],
         'snapshot_commit_sha':provenance['snapshot_commit_sha'],'notebook_sha256':staging['staged_notebook_sha256'],
         'verified_archive_members':len(archive['artifacts']),'precision_example_records':count,
         'metric_observations':len(observations),'environment':env['pip_consistency_state'],
         'memory':result['memory'],'artifact_manifests_verified':True,'evaluation_hashes_recomputed':True,
         'derived_metrics_recomputed':True,'packed_bytes_local_recomputed':False,
         'packed_bytes_note':'Hashes computed over actual CUDA model state in recorded runtime; packed checkpoints were not persisted and cannot be rehashed locally.',
         'attack_detected':None,'durability_pass':None,'gate_RG005':'BLOCKED',
         'supported_claims':result['supported_claims'],'unsupported_claims':result['unsupported_claims']}
print(json.dumps(verdict))
