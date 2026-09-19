import json
import urllib.parse
import urllib.request

queries = [
    "yolo drone detection",
    "air defense yolo",
    "military aircraft yolo",
    "tank yolo uav",
    "thermal person yolo",
    "visdrone yolo",
]
seen = set()
for q in queries:
    url = "https://www.kaggle.com/api/v1/datasets/list?search=" + urllib.parse.quote(q) + "&pageSize=8"
    req = urllib.request.Request(url, headers={"User-Agent": "tacyolo"})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    for d in data:
        ref = d.get("ref")
        if ref in seen:
            continue
        seen.add(ref)
        mb = round((d.get("totalBytes") or 0) / 1e6, 1)
        print(
            f"{mb:8.1f}MB  votes={d.get('voteCount'):4}  dl={d.get('downloadCount'):6}  {ref}  | {d.get('title')}"
        )
