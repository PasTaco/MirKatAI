class LiteratureTools:
    def __init__(self, chunks, supports, answer):
        self.chunks = chunks
        self.supports = supports
        self.answer = answer

    def process_chunk(self, chunk, n):
        title = chunk.web.title
        uri = chunk.web.uri
        return f"{n}. [{title}]({uri})"

    def create_bibliography(self):
        if self.chunks is not None:
            references = [self.process_chunk(chunk, i + 1) for i, chunk in enumerate(self.chunks)]
            return "\n".join(references)
        return ""

    def process_indices(self, indices):
        links = [f"[{i + 1}]({self.chunks[i].web.uri})" for i in indices]
        return f"[{','.join(links)}]"

    def process_text_ref(self, support):
        indices = support.grounding_chunk_indices
        text = support.segment.text
        return text, text + self.process_indices(indices)

    def process_references(self):
        modified_answer = self.answer
        if self.supports is not None:
            for support in self.supports:
                text, ref = self.process_text_ref(support)
                modified_answer = modified_answer.replace(text, ref)
        return modified_answer

    def process_paragraph(self):
        processed_answer = self.process_references()
        bibliography = self.create_bibliography()
        return f"**Answer**\n{processed_answer}\n\n**Bibliography**\n{bibliography}"
