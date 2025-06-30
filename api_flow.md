 curl -X 'POST'   'http://localhost:8001/api/v1/rank?stream=true'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
    "prompt": "Rank each of these top-level domains (descending order, where most relevant is first) according to their relevancy to the concept of math",
    "objects": [
  { "value": "edu" },
  { "value": "university" },
  { "value": "academy" }
    ],
    "config": {
      "batch_size": 10,
      "num_runs": 10,
      "token_limit": 128000,
      "refinement_ratio": 0.5,
      "encoding": "o200k_base",
      "provider": "openai",
      "openai_model": "gpt-4o-mini",
      "openrouter_model": "string",
      "openai_api_key": "sk-proj-",
      "openai_base_url": "https://api.openai.com/v1",
      "openrouter_api_key": "string",
      "openrouter_base_url": "https://openrouter.ai/api/v1",
      "template": "string",
      "dry_run": false
    }
  }'
{"event_type":"status","status":"starting","message":"Starting 10 ranking runs for 3 objects."}
{"event_type":"progress","run_number":1,"message":"Run 1/10 completed.","intermediate_results":[{"key":"5hO7RCvw","value":"edu","metadata":null,"score":1.0,"exposure":1,"rank":1},{"key":"5vkz92Jn","value":"academy","metadata":null,"score":2.0,"exposure":1,"rank":2},{"key":"rbpsDsio","value":"university","metadata":null,"score":3.0,"exposure":1,"rank":3}],"processing_time_current_run_ms":1046}
{"event_type":"progress","run_number":2,"message":"Run 2/10 completed.","intermediate_results":[{"key":"5hO7RCvw","value":"edu","metadata":null,"score":1.0,"exposure":2,"rank":1},{"key":"rbpsDsio","value":"university","metadata":null,"score":2.5,"exposure":2,"rank":2},{"key":"5vkz92Jn","value":"academy","metadata":null,"score":2.5,"exposure":2,"rank":3}],"processing_time_current_run_ms":1540}
{"event_type":"progress","run_number":3,"message":"Run 3/10 completed.","intermediate_results":[{"key":"5hO7RCvw","value":"edu","metadata":null,"score":1.0,"exposure":3,"rank":1},{"key":"rbpsDsio","value":"university","metadata":null,"score":2.3333333333333335,"exposure":3,"rank":2},{"key":"5vkz92Jn","value":"academy","metadata":null,"score":2.6666666666666665,"exposure":3,"rank":3}],"processing_time_current_run_ms":1085}
{"event_type":"progress","run_number":4,"message":"Run 4/10 completed.","intermediate_results":[{"key":"5hO7RCvw","value":"edu","metadata":null,"score":1.0,"exposure":4,"rank":1},{"key":"rbpsDsio","value":"university","metadata":null,"score":2.25,"exposure":4,"rank":2},{"key":"5vkz92Jn","value":"academy","metadata":null,"score":2.75,"exposure":4,"rank":3}],"processing_time_current_run_ms":1010}
{"event_type":"progress","run_number":5,"message":"Run 5/10 completed.","intermediate_results":[{"key":"5hO7RCvw","value":"edu","metadata":null,"score":1.0,"exposure":5,"rank":1},{"key":"rbpsDsio","value":"university","metadata":null,"score":2.4,"exposure":5,"rank":2},{"key":"5vkz92Jn","value":"academy","metadata":null,"score":2.6,"exposure":5,"rank":3}],"processing_time_current_run_ms":1101}
