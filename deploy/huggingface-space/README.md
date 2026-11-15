---
title: slotserve
emoji: 🚌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Live demo of continuous batching for LLM inference
---

# slotserve demo

A live demo of [slotserve](https://github.com/aryan2006-singh/slotserve), an
LLM inference server with token-level continuous batching. It runs
SmolLM2-135M-Instruct on the free CPU tier.

Open the app, press **Send 8 test requests at once**, and watch the slot bars
in the sidebar: short requests finish and hand their slot to a queued request
while long ones keep running.

The server is OpenAI-compatible, so you can also call it directly:

```bash
curl https://<your-space>.hf.space/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt": "Continuous batching is", "max_tokens": 32}'
```
