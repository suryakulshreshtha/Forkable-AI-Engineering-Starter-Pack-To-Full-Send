# Agents Without Magic

An agent is a loop: ask the model, run the tool it asked for, show it the result, repeat until it answers or you stop it. The hard part is not the loop, it is everything around it: validating arguments, surviving tool failures, refusing to spin forever, and not burning the budget on retries. You use a scripted fake model here, so every run is free, fast and repeatable.
