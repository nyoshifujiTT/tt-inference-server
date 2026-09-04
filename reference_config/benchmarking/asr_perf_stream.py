#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Streaming TTFT/TPOT/decode-TPS probe for ASR /v1/audio/transcriptions (stream=true, SSE).
# Client-side timing: first chunk arrival = TTFT; inter-chunk gaps = TPOT.
import sys, time, json, uuid, statistics
import urllib.request
import concurrent.futures as cf

HOST=sys.argv[1] if len(sys.argv)>1 else "http://127.0.0.1:8100"
MODEL=sys.argv[2] if len(sys.argv)>2 else "Qwen3-ASR-1.7B"
WAV=sys.argv[3] if len(sys.argv)>3 else "/home/ubuntu/ttwork/real_ja.wav"
N=int(sys.argv[4]) if len(sys.argv)>4 else 40
C=int(sys.argv[5]) if len(sys.argv)>5 else 4
MAXTOK=int(sys.argv[6]) if len(sys.argv)>6 else 100
body=open(WAV,"rb").read()

def one(_i):
    b=f"----s{uuid.uuid4().hex}"; parts=[]
    for k,v in {"model":MODEL,"temperature":0,"language":"ja","to_language":"ja","max_completion_tokens":MAXTOK,"stream":"true"}.items():
        parts+= [("--"+b).encode(), ('Content-Disposition: form-data; name="%s"'%k).encode(), b"", str(v).encode()]
    parts+= [("--"+b).encode(), b'Content-Disposition: form-data; name="file"; filename="a.wav"', b"Content-Type: audio/wav", b"", body, ("--"+b+"--").encode(), b""]
    data=b"\r\n".join(parts)
    req=urllib.request.Request(HOST+"/v1/audio/transcriptions",data=data,headers={"Content-Type":"multipart/form-data; boundary=%s"%b},method="POST")
    t0=time.perf_counter(); ttft=None; chunk_times=[]; ntok=0
    try:
        resp=urllib.request.urlopen(req,timeout=180)
        for raw in resp:
            line=raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            payload=line[5:].strip()
            if payload=="[DONE]": break
            now=time.perf_counter()
            try:
                j=json.loads(payload)
                delta=j.get("choices",[{}])[0].get("delta",{}).get("content","")
            except: delta=""
            if ttft is None: ttft=now-t0
            chunk_times.append(now); ntok+=1
        e2e=time.perf_counter()-t0
        # TPOT = mean gap between chunks after the first
        tpots=[chunk_times[i]-chunk_times[i-1] for i in range(1,len(chunk_times))]
        return {"ok":True,"ttft":ttft,"e2e":e2e,"ntok":ntok,"tpot_mean":statistics.mean(tpots) if tpots else None}
    except Exception as ex:
        return {"ok":False,"err":str(ex)[:60]}

for _ in range(3): one(0)  # warmup
t0=time.perf_counter(); res=[]
with cf.ThreadPoolExecutor(max_workers=C) as ex:
    for r in ex.map(one, range(N)): res.append(r)
wall=time.perf_counter()-t0
ok=[r for r in res if r.get("ok")]
def m(key): 
    xs=[r[key] for r in ok if r.get(key) is not None]; return round(statistics.mean(xs),3) if xs else None
def p(key,q):
    xs=sorted(r[key] for r in ok if r.get(key) is not None); return round(xs[min(len(xs)-1,int(q*len(xs)))],3) if xs else None
tot_tok=sum(r["ntok"] for r in ok)
rep={"concurrency":C,"requests":N,"ok":len(ok),"wall_s":round(wall,2),
 "mean_ttft_s":m("ttft"),"p99_ttft_s":p("ttft",0.99),
 "mean_e2e_s":m("e2e"),"p99_e2e_s":p("e2e",0.99),
 "mean_tpot_s":m("tpot_mean"),
 "decode_tps_per_user":round(1.0/m("tpot_mean"),2) if m("tpot_mean") else None,
 "decode_tps_aggregate":round(tot_tok/wall,2),
 "mean_tok_per_req":round(tot_tok/len(ok),1) if ok else None}
print("STREAMPERF "+json.dumps(rep))
