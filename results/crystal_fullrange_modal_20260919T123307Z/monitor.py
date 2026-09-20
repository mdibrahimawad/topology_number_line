"""Read-only Modal function statistics; never starts or changes an app."""
import asyncio
from datetime import datetime, timezone
import json
import sys
from modal.client import _Client
from modal_proto import api_pb2

async def main():
    c=await _Client.from_env()
    app_id=sys.argv[1]
    response=await c.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    functions=dict(response.app_layout.function_ids)
    seen=False
    empty=0
    for _ in range(360):
        values={}
        for name,fid in functions.items():
            r=await c.stub.FunctionGetCurrentStats(api_pb2.FunctionGetCurrentStatsRequest(function_id=fid))
            values[name]=dict(function_id=fid,running_inputs=r.num_running_inputs,containers=r.num_total_tasks,backlog=r.backlog)
        print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),app_id=app_id,functions=values)),flush=True)
        active=any(v['running_inputs'] or v['backlog'] for v in values.values())
        seen=seen or active
        empty=empty+1 if not active else 0
        if seen and empty>=2: break
        await asyncio.sleep(30)

asyncio.run(main())
