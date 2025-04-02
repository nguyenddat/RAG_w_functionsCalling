import os

import openai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key = os.getenv("OPENAI_API_KEY"),
    dimensions = 1024
)

llm = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

def get_prompt_template(task):
    if task == "rag":
        prompt_template = PromptTemplate.from_template(
            """
            Bạn là một trợ lý ảo của ban tuyển sinh một trường đại học. Nhiệm vụ của bạn là cung cấp và trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.
            Lưu ý:
            - Bạn chỉ dựa vào các thông tin cung cấp để trả lời câu hỏi.
            - Nếu không thể trả lời câu hỏi dựa trên thông tin cung cấp, bạn cần trả lời là tôi không có đủ thông tin để trả lời câu hỏi trên!
            
            Thông tin cung cấp:
            {context}

            Question:
            {question}
            """
        )
    
    else:
        prompt_template = PromptTemplate.from_template(
            """
            Bạn là một trợ lý ảo thông minh. Nhiệm vụ của bạn là chọn function phù hợp nhất để xử lý câu hỏi của người dùng.
            Dưới đây là danh sách các function có sẵn và mô tả của chúng:
            {functions_description}

            Câu hỏi:
            {question}

            Dựa trên câu hỏi của người dùng, bạn cần chọn function phù hợp nhất.
            Trả lời chỉ với tên function mà bạn chọn, không cần giải thích gì thêm.
            """
        )
    
    return prompt_template

def get_chat_completion(message: str, task: str, params = {}):
    formatted_prompt = get_prompt_template(task).format(**params)
    
    response = llm.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": message
            },
            {
                "role": "user",
                "content": formatted_prompt
            },
        ],
        temperature = 0
    )

    return response.choices[0].message.content

