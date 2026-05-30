from typing import Sequence

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
    query: str

def call_llm_1(state: State) -> dict:
    """Recebe o código e retorna a análise do modelo Gemini"""

    messages: Sequence[BaseMessage] = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f'Analise o seguinte código e forneça sugestões de melhorias:\n\n{state["query"]}')
    ]

    response: AIMessage = models['gemini-2.5-flash-lite'].invoke(messages)
    return {'llm1': response.content}

def call_llm_2(state: State) -> dict:
    """Recebe o código e retorna a análise do modelo Gemini"""

    messages: Sequence[BaseMessage] = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f'Analise o seguinte código e forneça sugestões de melhorias:\n\n{state["query"]}')
    ]

    response: AIMessage = models['gemini-2.5-flash-lite'].invoke(messages)
    return {'llm2': response.content}

def judge(state: State) -> dict:
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
    
    [Análise do Especialista A]
    {state['llm1']}
    
    [Análise do Especialista B]
    {state['llm2']}
    
    Forneça sua avaliação comparativa e conclua com seu veredito
    final usando exatamente um destes formatos:
    '[[A]] se a análise A for melhor'
    '[[B]] se a análise B for melhor'
    '[[C]] em caso de empate'
    """
    messages: Sequence[SystemMessage] = [SystemMessage(content=msg)]
    response: AIMessage = models['gemini-2.5-flash-lite'].invoke(messages)
    return {"best_llm": response.content}

code_analysis_builder = StateGraph(State)

code_analysis_builder.add_node("call_llm_1", call_llm_1)
code_analysis_builder.add_node("call_llm_2", call_llm_2)
code_analysis_builder.add_node("judge", judge)

code_analysis_builder.add_edge(START, "call_llm_1")
code_analysis_builder.add_edge(START, "call_llm_2")
code_analysis_builder.add_edge("call_llm_1", "judge")
code_analysis_builder.add_edge("call_llm_2", "judge")
code_analysis_builder.add_edge("judge", END)

code_analysis_workflow = code_analysis_builder.compile()