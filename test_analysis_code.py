from parallelization import code_analysis_workflow

codigo_teste: str = """
def calcular_media(lista):
    soma = 0
    for i in range(len(lista)):
        soma = soma + lista[i]
    media = soma / len(lista)
    return media

numeros = [1, 2, 3, 4, 5]
resultado = calcular_media(numeros)
print(f'A média é: {resultado}')
"""

resultado = code_analysis_workflow.invoke({
    'query': codigo_teste
})

print("\n=== Análise do Gemini 1 ===")
print(resultado["llm1"])
print("\n=== Análise do Gemini 2 ===")
print(resultado["llm2"])
print("\n=== Avaliação Final ===")
print(resultado["best_llm"])