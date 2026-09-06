#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Streaming TTFT/TPOT/decode-TPS probe for ASR /v1/audio/transcriptions (stream=true, SSE).
# Client-side timing: first chunk arrival = TTFT; inter-chunk gaps = TPOT.
#
# A chunk is NOT a token. vLLM's speech_to_text stream generator
# (entrypoints/speech_to_text/base/serving.py) yields one frame per non-empty
# post-processed delta, and with stream_options.include_usage it appends a
# usage-only frame carrying choices=[]. Counting frames therefore both
# undercounts (a delta may carry several tokens) and overcounts (the usage
# frame). Ask for the usage frame and take completion_tokens from it, which is
# the same quantity vllm:generation_tokens_total gives the non-streaming probe,
# so the two probes' decode-TPS columns are comparable. TPOT stays per frame --
# it is a frame-arrival gap by definition -- and is scaled by the measured
# tokens/frame so decode_tps_per_user is in tokens, not frames.
import sys, time, json, uuid, statistics
import urllib.request
import concurrent.futures as cf

# Same defaults as asr_perf_probe.py -- the two are meant to be run back to back
# on the same workload, so a different port, a different served name or a
# different request count between them silently breaks the comparison.
HOST=sys.argv[1] if len(sys.argv)>1 else "http://127.0.0.1:8110"
MODEL=sys.argv[2] if len(sys.argv)>2 else "neosophie/Qwen3-ASR-1.7B-JA"
if len(sys.argv)<=3:
    sys.exit("usage: %s [host] [model] <wav> [requests] [concurrency] [max_tokens]\n"
             "the clip is required; see the runbook's 'The clip to check with'"%sys.argv[0])
WAV=sys.argv[3]
N=int(sys.argv[4]) if len(sys.argv)>4 else 60
C=int(sys.argv[5]) if len(sys.argv)>5 else 4
MAXTOK=int(sys.argv[6]) if len(sys.argv)>6 else 100
body=open(WAV,"rb").read()

def one(_i):
    b=f"----s{uuid.uuid4().hex}"; parts=[]
    fields={"model":MODEL,"temperature":0,"language":"ja","to_language":"ja","max_completion_tokens":MAXTOK,"stream":"true",
            "stream_include_usage":"true"}
    for k,v in fields.items():
        parts+= [("--"+b).encode(), ('Content-Disposition: form-data; name="%s"'%k).encode(), b"", str(v).encode()]
    parts+= [("--"+b).encode(), b'Content-Disposition: form-data; name="file"; filename="a.wav"', b"Content-Type: audio/wav", b"", body, ("--"+b+"--").encode(), b""]
    data=b"\r\n".join(parts)
    req=urllib.request.Request(HOST+"/v1/audio/transcriptions",data=data,headers={"Content-Type":"multipart/form-data; boundary=%s"%b},method="POST")
    t0=time.perf_counter(); ttft=None; chunk_times=[]; nframe=0; ntok=None
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
            except Exception:
                continue
            usage=j.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                # usage-only frame: authoritative token count, carries no delta
                ntok=int(usage["completion_tokens"])
            choices=j.get("choices") or []
            if not choices:
                continue
            if not choices[0].get("delta",{}).get("content"):
                continue
            if ttft is None: ttft=now-t0
            chunk_times.append(now); nframe+=1
        e2e=time.perf_counter()-t0
        # TPOT = mean gap between content frames after the first
        tpots=[chunk_times[i]-chunk_times[i-1] for i in range(1,len(chunk_times))]
        return {"ok":True,"ttft":ttft,"e2e":e2e,"nframe":nframe,"ntok":ntok,
                "tpot_mean":statistics.mean(tpots) if tpots else None}
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
missing=[r for r in ok if r.get("ntok") is None]
if missing:
    # Without the usage frame there is no token count, only a frame count. Say
    # so rather than silently reporting frames per second as tokens per second.
    print("STREAMPERF "+json.dumps({"error":"server sent no usage frame; "
          "cannot report token-based throughput","requests":N,"ok":len(ok),
          "without_usage":len(missing)}))
    raise SystemExit(1)
tot_tok=sum(r["ntok"] for r in ok)
tot_frame=sum(r["nframe"] for r in ok)
tok_per_frame=(tot_tok/tot_frame) if tot_frame else None
mean_tpot=m("tpot_mean")
rep={"concurrency":C,"requests":N,"ok":len(ok),"wall_s":round(wall,2),
 "mean_ttft_s":m("ttft"),"p99_ttft_s":p("ttft",0.99),
 "mean_e2e_s":m("e2e"),"p99_e2e_s":p("e2e",0.99),
 "mean_tpot_s":mean_tpot,
 "tokens_per_frame":round(tok_per_frame,3) if tok_per_frame else None,
 "decode_tps_per_user":round(tok_per_frame/mean_tpot,2) if (mean_tpot and tok_per_frame) else None,
 "decode_tps_aggregate":round(tot_tok/wall,2),
 "mean_tok_per_req":round(tot_tok/len(ok),1) if ok else None,
 "mean_frames_per_req":round(tot_frame/len(ok),1) if ok else None}
print("STREAMPERF "+json.dumps(rep))
