"""Four parallel L40S containers render matched sampler pairs; user prompts stay local."""

import modal
from pathlib import Path
from images_app import moss_image,moss_cache,CACHE_DIR
app=modal.App('compare-moss-parallel')

@app.function(image=moss_image.add_local_file('images_app.py',remote_path='/root/images_app.py'),gpu='L40S',memory=65536,volumes={CACHE_DIR:moss_cache},max_containers=4,timeout=1200)
def compare_pair(prompt,negative,pair,seed):
    import time,base64,torch
    from moss_loader import MossBackend
    gpu=torch.cuda.get_device_name(0)
    backend=MossBackend(CACHE_DIR)
    results=[]
    for sampler in ['official_euler','er_sde']:
        start=time.monotonic()
        b64=backend.generate(prompt,720,1280,sampler=sampler,seed=seed,negative_prompt=negative)[0]
        seconds=round(time.monotonic()-start,2)
        print(f'PAIR {pair} {sampler} seed={seed} seconds={seconds} GPU={gpu}',flush=True)
        results.append({'pair':pair,'seed':seed,'sampler':sampler,'seconds':seconds,'gpu':gpu,'png':base64.b64decode(b64)})
    return results

@app.local_entrypoint()
def main(prompt_file: str, negative_file: str, output_dir: str = "outputs/sampler-comparison"):
    import json
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records=[]
    prompt=Path(prompt_file).read_text()
    negative=Path(negative_file).read_text().strip()
    args=[(prompt,negative,pair,20261022+pair) for pair in [1,2,3,4]]
    for results in compare_pair.starmap(args,order_outputs=False):
        for result in results:
            image=result.pop('png');name=f"pair-{result['pair']}-{result['sampler']}.png"
            (output/name).write_bytes(image)
            records.append({**result,'file':name})
            print('Saved',name,flush=True)
        (output/'parallel-results.json').write_text(json.dumps(records,indent=2))
