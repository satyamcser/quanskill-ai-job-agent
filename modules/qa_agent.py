from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class LocalLLM:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    def summarize_text(self, text: str, max_chars: int = 3000) -> str:
        """
        Summarize long job descriptions so they fit within token limits.
        """
        # Truncate long text before summarizing
        truncated_text = text[:max_chars]
        prompt = f"Summarize this job description in a concise way:\n\n{truncated_text}"
        summary = self.pipe(prompt, max_length=200, do_sample=False)
        return summary[0]["generated_text"]

    def answer_question(self, job_description: str, question: str) -> str:
        """
        Step 1: Summarize if needed.
        Step 2: Ask question on the summarized version.
        """
        #  If job desc is too long, summarize first
        if len(job_description.split()) > 200:
            summarized = self.summarize_text(job_description)
        else:
            summarized = job_description

        #  Ask question on summarized text
        prompt = (
            f"Based on the following summarized job description:\n\n"
            f"{summarized}\n\n"
            f"Answer this question: {question}"
        )

        response = self.pipe(prompt, max_length=256, do_sample=False)
        return response[0]["generated_text"]
