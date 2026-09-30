author:	Niko1221
association:	owner
edited:	false
status:	none
--
Thanks, that's a fair point. **Fixed in 0.1.38**: when a reply reaches max tokens while it is still thinking, the server window now says so and points to `reasoning_budget_tokens`. The budget stays opt-in, because a default budget would change answers for everyone. If your clients ask for high or xhigh, `"reasoning_budget_tokens": N` in `strata-<model>.json` sets it for every request (a request's own value still wins).
--
