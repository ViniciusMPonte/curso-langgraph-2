from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from models import models

SYSTEM_MESSAGE_LLMS: SystemMessage = SystemMessage(content="""
Você é um especialista em análise de código e boas
práticas de programação.
Sua tarefa é analisar o código fornecido e sugerir
melhorias em termos de:
1. Performance e otimização
2. Boas práticas e padrões de código
3. Segurança e tratamento de erros
4. Legibilidade e manutenibilidade

Forneça suas sugestões de forma estruturada e clara,
com exemplos práticos de como implementar as melhorias sugeridas.
Seja especifico e detalhado em suas recomendações.
""")

class State(TypedDict):
    llm1: str
    llm2: str
    best_llm: str

def call_llm_1(state: State) -> dict[str, AIMessage]:
    """Recebe o código e retorna a análise do modelo Gemini"""

    messages: Sequence[BaseMessage] = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f'Analise o seguinte código e forneça sugestões de melhorias:\n\n{state["query"]}')
    ]

    response: AIMessage = models['gemini-2.5-flash-lite'].invoke(messages)
    return {'llm1': response.content}

def call_llm_2(state: State) -> dict[str, AIMessage]:
    """Recebe o código e retorna a análise do modelo Gemini"""

    messages: Sequence[BaseMessage] = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f'Analise o seguinte código e forneça sugestões de melhorias:\n\n{state["query"]}')
    ]

    response: AIMessage = models['gemini-2.5-flash-lite'].invoke(messages)
    return {'llm2': response.content}

def judge(state: State):
    """Avalia qual análise foi mais completa e útil"""
    msg = f"""
    Aja como revisor técnico sênior e avalie a quantidade das
    análises de código fornecidas por dois especialistas.
    
    Sua tarefa é escolher a análise que:
    1. Identifica mais problemas potenciais.
    2. Fornece sugestões mais práticas e implementáveis.
    3. Considera aspectos do código, como perfomance, segurança, legibilidade, etc.
    4. Explica melhor o raciocínio por trás das sugestões.
    
    [Código Analisado]
    {state['query']}
    """