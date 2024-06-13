import os
from raptor import RetrievalAugmentation 

os.environ["OPENAI_API_KEY"] = ""



with open('demo/sample.txt', 'r') as file:
    text = file.read()
print(text[:100])
RA = RetrievalAugmentation()
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"

answer = RA.answer_question(question=question)

print("Answer: ", answer)

RA.add_to_existing("Cinderalla suffered from a severe illness, which was a stage 4 form of cancer.")