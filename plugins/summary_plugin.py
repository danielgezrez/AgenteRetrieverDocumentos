
class SummaryPlugin:
    def __init__(self, kernel):
        self.kernel = kernel

    async def summarize(self, text: str) -> str:
        prompt = f"""
        Resume el siguiente texto de forma clara, breve y sin perder información importante.
        
        Texto:
        {text}
        """
        result = await self.kernel.invoke_prompt(prompt)
        return str(result)