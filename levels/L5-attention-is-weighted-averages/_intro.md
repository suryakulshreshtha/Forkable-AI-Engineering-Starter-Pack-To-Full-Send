# Attention Is Weighted Averages

Strip away the diagrams and a transformer layer does one thing: each position takes a weighted average of every other position, with weights from a softmax over similarity scores. You build a softmax that cannot overflow, scaled dot-product attention with a causal mask, and the decoding knobs (temperature, top-k, top-p) that decide what a model actually says.
