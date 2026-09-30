author:	Interpause
association:	none
edited:	true
status:	none
--
dupe of #210, in fact #210 is likely the root cause. what youre suggesting is to report when it happens better
--
author:	alphastorm
association:	contributor
edited:	false
status:	none
--
Thanks, you're right: #210 covers this. Its second trigger is this case (the turn ends inside a call and `_close_scan` closes it), and its fail-loud fix is what this issue asks for. #210 went up a minute before this one, so our duplicate search missed it. Closing as a duplicate of #210.

One detail for that fix: the OpenAI stream counts a call as soon as its name is streamed ([server.py#L1072-L1074](https://github.com/Niko1221/Strata/blob/a79080535d1b2a71a3419a0d97d8e7dca194b0f1/serve/server.py#L1072-L1074)), and [L1087](https://github.com/Niko1221/Strata/blob/a79080535d1b2a71a3419a0d97d8e7dca194b0f1/serve/server.py#L1087) turns `stop` plus any counted call into `finish_reason: "tool_calls"`; the non-streamed answer takes its finish reason from the same stream. So a fix in `finish()` / `_close_scan()` alone would still report `tool_calls`; the finish reason also has to follow whether the call was completed.
--
author:	alphastorm
association:	contributor
edited:	false
status:	none
--
Reopening: 0.1.28 fixed #210's first trigger (`</parameter>` / `</tool_call>` inside a value), not this one. When the output ends inside a call, `finish()` ([frontend.py#L532-L535](https://github.com/Niko1221/Strata/blob/bbaaabb4643bb7873d4cef9d48b5dcf96e6cbff4/serve/frontend.py#L532-L535)) still closes it and emits `tool_call`, and the answer still ends with `finish_reason: "tool_calls"`.

At `bbaaabb` (0.1.28), no model:

```python
from serve.frontend import OutputParser

body = ("<tool_call>\n<function=write>\n"
        "<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=content>\npartial content")  # the turn ends here
p = OutputParser(thinking=False, stream_tools=True)
for ev in p.feed(body) + p.finish():
    print(ev.kind, "|", ev.text)
```

```
tool_start |
tool_args | {
tool_args | "path":
tool_args | "a.txt"
tool_args | ,"content":
tool_args | "partial content"}
tool_call |
```

Through `python -m serve.server --engine mock --script "</think>\n\n<the text above>"`, a streamed and a non-streamed OpenAI request with a `write` tool both end with `finish_reason: "tool_calls"` and the arguments `{"path":"a.txt","content":"partial content"}`. In the web app's MCP loop, the unfinished call runs with empty arguments.

#231 proposes a fix.

--
author:	Niko1221
association:	owner
edited:	false
status:	none
--
Confirmed. #231 goes into 0.1.31, plus one more change: a non-streamed OpenAI answer also drops tool calls whose arguments don't parse when the answer was cut, since many SDKs run any tool calls present.
--
author:	Niko1221
association:	owner
edited:	false
status:	none
--
Fixed in [0.1.31](https://github.com/Niko1221/Strata/releases/tag/v0.1.31) (#231 plus the non-streamed follow-up). Thanks!
--
