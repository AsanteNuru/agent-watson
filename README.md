The issue that I am running into currently:

After running the Agent gives a 200 and no further error except for a missing logfire.configure() which I am currently ignoring. 
  watson-1  | INFO:     Started server process [1]
  watson-1  | INFO:     Waiting for application startup.
  watson-1  | INFO:     Application startup complete.
  watson-1  | INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
  watson-1  | INFO:     172.18.0.1:61112 - "OPTIONS /api/watson_api_endpoint HTTP/1.1" 200 OK

Unfortunately I am getting no response from the Agent in the Agent 0 framework. 
When checking, the messages are received and stored in the supabase table. 
In that table the Agent also responds - when looking at the logs there are a bunch of errors. 
Thank you for all help. 
