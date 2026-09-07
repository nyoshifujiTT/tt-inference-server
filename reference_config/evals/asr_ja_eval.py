#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Japanese ASR eval+benchmark client for tt-inference-server vLLM /v1/audio/transcriptions.
# Sends local wav clips with customer(gbase-asr)-preset params, computes CER/WER (jiwer)
# and aggregate latency/RTF. Datasets: TEDxJP-10K (monologue) & MagicHub JA conversation.
import argparse, io, json, os, re, sys, time, uuid, wave, unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

USER_AGENT = "tt-asr-ja-eval/1.0"

def eprint(m): sys.stderr.write(m+"\n")

def wav_dur(path):
    with wave.open(path,'rb') as w:
        return w.getnframes()/float(w.getframerate())

def multipart(fields, fname, fbytes):
    b=f"----ttja{uuid.uuid4().hex}"; L=[]
    for k,v in fields.items():
        L+=[f"--{b}".encode(), f'Content-Disposition: form-data; name="{k}"'.encode(), b"", str(v).encode()]
    L+=[f"--{b}".encode(), f'Content-Disposition: form-data; name="file"; filename="{fname}"'.encode(),
        b"Content-Type: audio/wav", b"", fbytes, f"--{b}--".encode(), b""]
    return b"\r\n".join(L), b

def resolve_secs(resp, measured):
    """Audio duration to score speed against, in seconds.

    Prefer the duration WE measured from the file we submitted. The server's own
    figure is a billing quantity, not a measurement: OpenAI's usage.seconds is
    whole seconds, so it rounds every clip up. Trusting it inflated the TED-509
    total from the true 1649.4 s to 1892.0 s (+14.7%) and flattered
    throughput_audio_per_s and rtf_sum_lat_over_audio by the same factor, while
    the per-clip durations recorded alongside stayed correct -- so the error only
    showed up in the aggregate. Same rule as
    reference_config/benchmarking/asr_openai_benchmark.py.

    Fall back to the response only when the local file could not be measured.
    """
    if measured is not None:
        return measured
    d=resp.get("duration")
    if d is not None:
        try: return float(d)
        except: pass
    u=resp.get("usage")
    if isinstance(u,dict) and u.get("seconds") is not None:
        try: return float(u["seconds"])
        except: pass
    return None

def transcribe(host, fbytes, model, dur, timeout, resp_fmt):
    # customer gbase-asr preset
    fields={"model":model,"response_format":resp_fmt,"temperature":0,
            "language":"ja","to_language":"ja","repetition_penalty":1.1,
            "max_completion_tokens":int(dur*12)+32}
    body,bnd=multipart(fields,"audio.wav",fbytes)
    url=f"{host.rstrip('/')}/v1/audio/transcriptions"
    hdr={"Content-Type":f"multipart/form-data; boundary={bnd}","Accept":"application/json","User-Agent":USER_AGENT}
    req=urllib.request.Request(url,data=body,headers=hdr,method="POST")
    t0=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            payload=r.read().decode("utf-8"); ok=r.status==200
    except Exception as e:
        return False,time.perf_counter()-t0,None,f"ERROR: {e}"
    el=time.perf_counter()-t0
    try: data=json.loads(payload)
    except: return ok,el,None,payload[:200]
    # dur is what wav_dur() read off this very file; pass it, do not discard it
    return ok,el,resolve_secs(data,dur),data.get("text","")

# --- Japanese text normalization for CER/WER ---
def norm_ja(s):
    s=unicodedata.normalize("NFKC",s)
    s=re.sub(r"[\s、。,\.\?!？！「」『』（）\(\)\[\]【】…・:;：；\"'`~〜ー－\-]", "", s)
    return s.strip()

def cer(ref, hyp):
    r=list(norm_ja(ref)); h=list(norm_ja(hyp))
    return _edit(r,h)

def _edit(r,h):
    n,m=len(r),len(h)
    if n==0: return 0.0 if m==0 else 1.0
    dp=list(range(m+1))
    for i in range(1,n+1):
        prev=dp[0]; dp[0]=i
        for j in range(1,m+1):
            cur=dp[j]
            dp[j]=min(dp[j]+1, dp[j-1]+1, prev+(0 if r[i-1]==h[j-1] else 1))
            prev=cur
    return dp[m]/n

def load_manifest(path):
    # jsonl: {"wav":..., "ref":..., "id":...}
    items=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line: items.append(json.loads(line))
    return items

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="http://127.0.0.1:8110")
    ap.add_argument("--model",default="neosophie/Qwen3-ASR-1.7B-JA")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--concurrency",type=int,default=4)
    ap.add_argument("--timeout",type=int,default=120)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--response-format",default="json",choices=["json","verbose_json"])
    ap.add_argument("--output",default=None)
    ap.add_argument("--samples-out",default=None)
    a=ap.parse_args()
    items=load_manifest(a.manifest)
    if a.limit>0: items=items[:a.limit]
    # preload bytes + dur
    data=[]
    for it in items:
        try:
            with open(it["wav"],"rb") as f: fb=f.read()
            d=wav_dur(it["wav"])
        except Exception as e:
            eprint(f"skip {it.get('id')}: {e}"); continue
        data.append((it,fb,d))
    eprint(f"loaded {len(data)} clips, conc={a.concurrency}")
    results=[None]*len(data)
    def one(i):
        it,fb,d=data[i]
        ok,el,asec,txt=transcribe(a.host,fb,a.model,d,a.timeout,a.response_format)
        return i,ok,el,d,asec,txt,it
    t0=time.perf_counter(); done=0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs=[ex.submit(one,i) for i in range(len(data))]
        for fu in as_completed(futs):
            i,ok,el,d,asec,txt,it=fu.result()
            results[i]=(ok,el,d,asec,txt,it); done+=1
            if done%50==0: eprint(f"{done}/{len(data)}")
    wall=time.perf_counter()-t0
    lat=[]; tot_ref=0; tot_err=0; audio=0; okc=0; failc=0
    samples=[]
    for r in results:
        if r is None: continue
        ok,el,d,asec,txt,it=r
        if ok:
            okc+=1; lat.append(el); audio+=(asec or d)
            rn=norm_ja(it["ref"]); hn=norm_ja(txt)
            tot_ref+=len(rn); tot_err+=round(cer(it["ref"],txt)*max(1,len(rn)))
            samples.append({"id":it.get("id"),"dur":round(d,2),"lat":round(el,3),
                            "ref":it["ref"],"hyp":txt,"cer":round(cer(it["ref"],txt),4)})
        else:
            failc+=1; samples.append({"id":it.get("id"),"error":txt})
    agg_cer=tot_err/tot_ref if tot_ref else None
    def pct(x,q):
        if not x: return None
        xs=sorted(x); k=min(len(xs)-1,int(q*len(xs))); return round(xs[k],3)
    rep={"host":a.host,"model":a.model,"manifest":a.manifest,"clips":len(data),
         "ok":okc,"fail":failc,"concurrency":a.concurrency,
         "wall_s":round(wall,2),"audio_s":round(audio,1),
         "throughput_audio_per_s":round(audio/wall,3) if wall else None,
         "rtf_sum_lat_over_audio":round(sum(lat)/audio,4) if audio else None,
         "corpus_cer":round(agg_cer,4) if agg_cer is not None else None,
         "mean_lat_s":round(sum(lat)/len(lat),3) if lat else None,
         "p50_lat_s":pct(lat,0.5),"p90_lat_s":pct(lat,0.9),"p99_lat_s":pct(lat,0.99)}
    print(json.dumps(rep,ensure_ascii=False,indent=2))
    if a.output:
        with open(a.output,"w") as f: json.dump(rep,f,ensure_ascii=False,indent=2)
    if a.samples_out:
        with open(a.samples_out,"w") as f:
            for s in samples: f.write(json.dumps(s,ensure_ascii=False)+"\n")

if __name__=="__main__": main()
