"""
Interactive chat with local LLM and internet search capability.
Uses OpenAI-compatible API with function calling for web searches.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from search_tool import search_web, fetch_url, TOOLS_SCHEMA

# Load environment variables
load_dotenv()

# Configure OpenAI client for local LLM
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.20.18:12435/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "sk-no-key-required")
)

# System prompt that instructs LLM when to use search
SYSTEM_PROMPT = """Ты полезный ассистент с доступом к поиску в интернете.

ПРАВИЛА ИСПОЛЬЗОВАНИЯ ПОИСКА:
- Используй search_web, когда пользователь спрашивает о:
  - Текущих событиях, новостях, происшествиях
  - Актуальных ценах, курсах валют, котировках
  - Погоде, спортивных результатах
  - Последних версиях ПО, библиотек, фреймворков
  - Любых данных, которые могли измениться после твоего обучения

- НЕ используй поиск для:
  - Общих знаний, фактов, определений
  - Математических вычислений
  - Написания кода (если не нужны актуальные API/документация)
  - Философских вопросов, творческих задач

Когда используешь поиск:
1. Вызови search_web с точным запросом
2. Получив результаты, дай ответ со ссылками на источники
3. Будь краток и указывай только релевантную информацию

Отвечай на русском языке, если пользователь не указал иное."""

# Available functions for tool calling
AVAILABLE_FUNCTIONS = {
    "search_web": search_web,
    "fetch_url": fetch_url
}


def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a tool function by name."""
    if name not in AVAILABLE_FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}
    try:
        func = AVAILABLE_FUNCTIONS[name]
        if name == "search_web":
            return func(arguments.get("query", ""))
        elif name == "fetch_url":
            return func(arguments.get("url", ""))
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Invalid arguments"}


def format_search_results(results: dict) -> str:
    """Format search results for LLM context."""
    if "error" in results and not results.get("results"):
        return f"Search error: {results['error']}"
    
    formatted = []
    for i, r in enumerate(results.get("results", []), 1):
        formatted.append(
            f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['content']}"
        )
    return "\n\n".join(formatted) if formatted else "No results found."


def chat_loop():
    """Main interactive chat loop with search capability."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("=" * 60)
    print("Чат с локальной LLM + поиск в интернете")
    print("=" * 60)
    print("Команды: 'quit' - выход, 'clear' - очистить историю")
    print("Поиск активируется автоматически при необходимости")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "выход"):
                print("До свидания!")
                break
            
            if user_input.lower() in ("clear", "очистить"):
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("История очищена.\n")
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # First LLM call - may request tool use
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "local-model"),
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                stream=False
            )
            
            assistant_message = response.choices[0].message
            
            # Check if LLM wants to use a tool
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = eval(tool_call.function.arguments)
                    
                    print(f"\n🔍 Поиск: {func_args.get('query', func_args.get('url', ''))}")
                    
                    # Execute the tool
                    tool_result = execute_tool(func_name, func_args)
                    
                    # Format result for LLM
                    if func_name == "search_web":
                        formatted_result = format_search_results(tool_result)
                    else:
                        formatted_result = tool_result.get("content", str(tool_result))
                    
                    # Add tool call and result to messages
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [tool_call.to_dict()] if hasattr(tool_call, 'to_dict') else [tool_call]
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": formatted_result
                    })
                
                # Second LLM call - generate final response with search results
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "local-model"),
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                    stream=True
                )
                
                print("\nАссистент: ", end="", flush=True)
                final_message = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        final_message += content
                print()
                
                messages.append({"role": "assistant", "content": final_message})
                
            else:
                # No tool needed - stream response directly
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "local-model"),
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                    stream=True
                )
                
                print("\nАссистент: ", end="", flush=True)
                final_message = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        final_message += content
                print()
                
                messages.append({"role": "assistant", "content": final_message})
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nПрервано. До свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}")
            print("Проверьте подключение к LLM серверу и настройки API")
            break


if __name__ == "__main__":
    chat_loop()
