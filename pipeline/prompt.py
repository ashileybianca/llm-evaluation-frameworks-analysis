"""The only place that builds the prompt: one user message, no system message.

    Answer the question directly, using only the context.

    Context: {passage 1}, a blank line, {passage 2}, ...
    Question: {question}

The instruction is the same for every dataset. It asks for a direct answer (no preamble such as "According to the
text") and to stay in the context. It sets no length, so the length of the answer is measured later, not
controlled. It is not tuned per dataset, so it does not favor any.
Passages keep the dataset's original order, because Context Precision depends on the position of the relevant
ones. Changing this format changes the token counts and the answers: run characterize.py and generate.py again.
"""

CONTENT_TEMPLATE = "Answer the question directly, using only the context.\n\nContext: {context}\nQuestion: {question}\n"


def build_messages(sample):
    """Returns the chat messages for one question."""
    context = "\n\n".join(sample["contexts"])
    content = CONTENT_TEMPLATE.format(context=context, question=sample["question"])
    return [{"role": "user", "content": content}]
