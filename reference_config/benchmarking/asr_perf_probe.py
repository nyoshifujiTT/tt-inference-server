#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Measure E2E / TTFT / prefill / decode TPS / TPS-per-user for ASR via vLLM /metrics deltas.
# Fixed workload: same wav, fixed max_tokens, N requests at concurrency C.
import sys, time, json, urllib.request, re
import concurrent.futures as cf

# Defaults match the runbook's worked example. The wav has no default on
# purpose: the runbook tells you to fetch FLEURS ja_jp test[0] rather than copy
# a wav out of someone's scratch directory, and a default pointing at a
# bring-up host's ~/ttwork contradicted that in the one place that runs.
HOST=sys.argv[1] if len(sys.argv)>1 else "http://127.0.0.1:8110"
MODEL=sys.argv[2] if len(sys.argv)>2 else "neosophie/Qwen3-ASR-1.7B-JA"
if len(sys.argv)<=3:
    sys.exit("usage: %s [host] [model] <wav> [requests] [concurrency] [max_tokens]\n"
             "the clip is required; see the runbook's 'The clip to check with'"%sys.argv[0])
WAV=sys.argv[3]
N=int(sys.argv[4]) if len(sys.argv)>4 else 60
C=int(sys.argv[5]) if len(sys.argv)>5 else 4
MAXTOK=int(sys.argv[6]) if len(sys.argv)>6 else 100

def metrics():
    raw=urllib.request.urlopen(HOST+"/metrics",timeout=10).read().decode()
    d={}
    # Every figure this probe prints is a difference of two of these counters,
    # so a counter that stops matching does not produce an error: both reads
    # return the same substitute and the difference is a plausible-looking
    # zero. gen_tokens 0 / decode_tps_aggregate 0.0 would then be reported as
    # a measurement. Fail on the spot instead -- a renamed or relabelled vLLM
    # counter is a reason to fix the probe, not a result.
    def g(pat, name):
        m=re.search(pat,raw)
        if not m:
            sys.exit("probe: /metrics has no %s (pattern %r). vLLM's counter "
                     "set has changed; every figure here is a delta of these, "
                     "so continuing would report zeros as measurements."%(name,pat))
        return float(m.group(1))

    # findall gives [] rather than None when a counter goes away, and sum([])
    # is 0 -- the same silent substitution g() refuses.
    def gsum(pat, name):
        vals=re.findall(pat,raw)
        if not vals:
            sys.exit("probe: /metrics has no %s (pattern %r)."%(name,pat))
        return sum(float(x) for x in vals)
    d["ttft_sum"]=g(r'vllm:time_to_first_token_seconds_sum\{[^}]*\}\s+([0-9.eE+]+)',"vllm:time_to_first_token_seconds_sum")
    d["ttft_cnt"]=g(r'vllm:time_to_first_token_seconds_count\{[^}]*\}\s+([0-9.eE+]+)',"vllm:time_to_first_token_seconds_count")
    d["e2e_sum"]=g(r'vllm:e2e_request_latency_seconds_sum\{[^}]*\}\s+([0-9.eE+]+)',"vllm:e2e_request_latency_seconds_sum")
    d["e2e_cnt"]=g(r'vllm:e2e_request_latency_seconds_count\{[^}]*\}\s+([0-9.eE+]+)',"vllm:e2e_request_latency_seconds_count")
    d["pref_sum"]=g(r'vllm:request_prefill_time_seconds_sum\{[^}]*\}\s+([0-9.eE+]+)',"vllm:request_prefill_time_seconds_sum")
    d["dec_sum"]=g(r'vllm:request_decode_time_seconds_sum\{[^}]*\}\s+([0-9.eE+]+)',"vllm:request_decode_time_seconds_sum")
    d["gen_tok"]=g(r'vllm:generation_tokens_total\{[^}]*\}\s+([0-9.eE+]+)',"vllm:generation_tokens_total")
    # request_success total across finish reasons. findall gives [] rather
    # than None when the counter goes away, and sum([]) is 0 -- the same
    # silent substitution g() now refuses, so refuse it here too.
    d["succ"]=gsum(r'vllm:request_success_total\{[^}]*\}\s+([0-9.eE+]+)',"vllm:request_success_total")
    return d

body=open(WAV,"rb").read()
import uuid
def transcribe(_i):
    b=f"----p{uuid.uuid4().hex}"
    parts=[]
    for k,v in {"model":MODEL,"response_format":"json","temperature":0,"language":"ja","to_language":"ja","max_completion_tokens":MAXTOK}.items():
        parts.append(("--"+b).encode()); parts.append(('Content-Disposition: form-data; name="%s"'%k).encode()); parts.append(b""); parts.append(str(v).encode())
    parts.append(("--"+b).encode()); parts.append(b'Content-Disposition: form-data; name="file"; filename="a.wav"'); parts.append(b"Content-Type: audio/wav"); parts.append(b""); parts.append(body)
    parts.append(("--"+b+"--").encode()); parts.append(b"")
    data=b"\r\n".join(parts)
    req=urllib.request.Request(HOST+"/v1/audio/transcriptions",data=data,headers={"Content-Type":"multipart/form-data; boundary=%s"%b},method="POST")
    t=time.perf_counter()
    try:
        urllib.request.urlopen(req,timeout=180).read()
        return time.perf_counter()-t, True
    except Exception as e:
        return time.perf_counter()-t, False

# warmup 3
for _ in range(3): transcribe(0)
m0=metrics(); t0=time.perf_counter()
lat=[]; ok=0
with cf.ThreadPoolExecutor(max_workers=C) as ex:
    for l,s in ex.map(transcribe, range(N)):
        lat.append(l); ok+= 1 if s else 0
wall=time.perf_counter()-t0
m1=metrics()
d_cnt=m1["ttft_cnt"]-m0["ttft_cnt"]
d_gen=m1["gen_tok"]-m0["gen_tok"]
d_ttft=m1["ttft_sum"]-m0["ttft_sum"]
d_e2e=m1["e2e_sum"]-m0["e2e_sum"]
d_pref=m1["pref_sum"]-m0["pref_sum"]
d_dec=m1["dec_sum"]-m0["dec_sum"]
rep={
 "concurrency":C,"requests":N,"ok":ok,"wall_s":round(wall,2),
 "req_per_s":round(d_cnt/wall,3),
 "gen_tokens":int(d_gen),
 "decode_tps_aggregate":round(d_gen/wall,2),          # total generated tok / wall (system throughput)
 "decode_tps_per_user":round(d_gen/d_dec,2) if d_dec else None,  # tok / sum(decode time) = per-stream decode speed
 "mean_ttft_s":round(d_ttft/d_cnt,3) if d_cnt else None,
 "mean_e2e_s":round(d_e2e/d_cnt,3) if d_cnt else None,
 "mean_prefill_s":round(d_pref/d_cnt,3) if d_cnt else None,
 "mean_decode_s":round(d_dec/d_cnt,3) if d_cnt else None,
 "mean_gen_tok_per_req":round(d_gen/d_cnt,1) if d_cnt else None,
 "client_e2e_mean_s":round(sum(lat)/len(lat),3),
}
print("PERF "+json.dumps(rep))
