from semantic_kernel.functions import kernel_function

class DocumentPlugin:

    @kernel_function(
        name="filter_by_source",
        description="Filtra resultados por nombre de documento"
    )
    def filter_by_source(self, results, source_name: str):
        return [r for r in results if r["source"] == source_name]


    @kernel_function(
        name="summarize_context",
        description="Resume contexto recuperado antes de responder"
    )
    def summarize_context(self, texts: list):
        return "\n".join(texts[:5])