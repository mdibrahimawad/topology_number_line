"""Read-only Modal task/concurrency snapshots; never creates remote calls."""
import asyncio
import json
import sys
import time
from modal.client import _Client
from modal_proto import api_pb2


async def main():
    client = await _Client.from_env()
    app_id = sys.argv[1]
    while True:
        try:
            apps = await client.stub.AppList(api_pb2.AppListRequest(environment_name='main'))
            app = next(a for a in apps.apps if a.app_id == app_id)
            layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
            stats = {}
            for name, fid in layout.app_layout.function_ids.items():
                s = await client.stub.FunctionGetCurrentStats(api_pb2.FunctionGetCurrentStatsRequest(function_id=fid))
                stats[name] = dict(running=s.num_running_inputs, containers=s.num_total_tasks, backlog=s.backlog)
            print(json.dumps(dict(time=time.time(), app_id=app_id, state=app.state, functions=stats)), flush=True)
            if app.state == api_pb2.APP_STATE_STOPPED:
                break
        except Exception as exc:
            print(json.dumps(dict(time=time.time(), error=type(exc).__name__)), flush=True)
        await asyncio.sleep(20)


asyncio.run(main())
