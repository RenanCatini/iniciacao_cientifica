# Anotações - Árvore de Decisão 🌳

**Matéria:** Machine Learning
**Data:** 21/09/2025

---

### Ideia Principal
* É tipo um jogo de "Adivinhe Quem?".
* Faz perguntas (sim/não) pra ir eliminando as opções até sobrar uma só.

### O que é?
* Modelo de ML em formato de fluxograma / árvore.
* Toma decisões com base numa sequência de perguntas sobre os dados.
* Cada pergunta quebra os dados em grupos menores.

### Como funciona? (A parte importante)

**A grande dúvida:** Como a árvore escolhe a "melhor" pergunta pra fazer em cada etapa?

**Resposta: PUREZA de um grupo.**

* **Grupo Puro:** Todo mundo no grupo tem a mesma classificação/resposta.
    * Ex: Caixa só com maçãs.
    * É o nosso objetivo. Chegou num grupo puro = FIM. Vira uma **"folha"** da árvore.

* **Grupo Impuro (Misto):** As respostas estão misturadas.
    * Ex: Caixa com maçãs e laranjas.
    * PROBLEMA! Precisa fazer mais perguntas pra tentar separar.

**OBJETIVO DO ALGORITMO:** Fazer a pergunta que **mais aumenta a pureza** (ou mais diminui a impureza) dos grupos que vão ser criados.

* *Lembrete: Como ele mede a impureza? Usa umas fórmulas matemáticas tipo **Índice Gini** ou **Entropia**, que basicamente dão notas para o grupo, o quão impuro ou puro ele está.*

### Vantagens e Desvantagens

**👍 Vantagens:**
* Fácil de entender e visualizar. Dá pra desenhar.
* Não precisa de mto pré-processamento de dados.

**👎 Desvantagens:**
* **OVERFITTING!!! (MUITO CUIDADO AQUI)**
* O que é? A árvore fica complexa demais e "decora" os dados de treino.
* Ela não aprende a regra geral, só os exemplos específicos.
* **Analogia:** O aluno que decora as respostas da prova. Se mudar a pergunta, ele erra.

### Mão na Massa: O `DecisionTreeClassifier` do `sklearn` 🐍

Na prática, a gente não implementa isso do zero. Usamos a biblioteca `scikit-learn`.

```python
from sklearn.tree import DecisionTreeClassifier

# Criando o modelo com os parâmetros padrão
modelo_arvore = DecisionTreeClassifier()

# Depois é só treinar com os dados
# modelo_arvore.fit(X_treino, y_treino)
```

**Hiperparâmetros importantes na hora de criar o modelo:**

```python
    DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
```


* `criterion='gini'` (ou `'entropy')`
    * É aqui que a gente escolhe a "régua" pra medir a impureza. Lembra das 'notas'? É isso.
    * `'gini'` é o padrão e geralmente **mais rápido**, funciona calculando a probabilidade de errar (se tiver uma caixa com apenas 1 tipo, o gini=0, se tiver 2 tipos 50/50, o gini será de 0.5). Vai de 0 até 0.5 (para 2 classes).
        * Calcula a partir da multiplicação simples de probabilidades.
        * $$I_G(p) = 1 - \sum_{i=1}^{k} (p_i)^2$$
        * Onde:
            * **$k$**: É o número total de classes.
            * **$p_i$**: É a proporção (probabilidade) da classe *i* no grupo.
    * `'entropy'` é geralmente um pouco **mais lento**, funciona calculando o nível de 'surpresa' ou 'desordem' (se tiver uma caixa com apenas 1 tipo, a surpresa é 0, se tiver 2 tipos 50/50, a surpresa/incerteza é máxima = 1). Vai de 0 até 1.0 (para 2 classes).
        * Calcula usando logaritmos, que vem da Teoria da Informação.
        * $$H(p) = - \sum_{i=1}^{k} p_i \log_2(p_i)$$
        * Onde:
            * **$k$**: É o número total de classes.
            * **$p_i$**: É a proporção (probabilidade) da classe *i* no grupo.

* `max_depth=None` (ou um número, ex: `3`)
    * **Profundidade máxima** da árvore. Quantas "camadas" de perguntas ela pode fazer.
    * Se deixar `None` (padrão), ela cresce até não poder mais -> **GRANDE CHANCE DE OVERFITTING!**
        * **Por quê?** Sem limite, a árvore cria regras super específicas só pra garantir que cada folha seja 100% pura. Ela acaba **"decorando"** os dados de treino em vez de aprender o padrão geral.
    * É a principal ferramenta pra **"podar"** (limitar) a árvore.
    * *Dica: Começar com um valor baixo (3, 4, 5) é uma boa prática.*

* `min_samples_split=2` (ou um número maior)
    * **Número mínimo de amostras** que um nó precisa ter pra poder ser dividido (pra fazer uma nova pergunta).
    * Ajuda a evitar que a árvore crie regras pra grupos muito pequenos e específicos. -> Combate o overfitting.
        * **Por quê?** Isso impede que a árvore crie uma nova 'regra' (uma nova pergunta) baseada em pouquíssimos exemplos. Se aumentamos para `min_samples_split=20`, a árvore só vai se dar ao trabalho de criar uma nova divisão se tiver um grupo de, no mínimo, 20 amostras ("linhas"). Isso força o modelo a focar em padrões que aparecem em grupos maiores, que tendem a ser mais gerais e menos baseados em ruído.

* `min_samples_leaf=1` (ou um número maior)
    * **Número mínimo de amostras** que um nó final (uma "folha") precisa ter.
    * Garante que nossas respostas não sejam baseadas em um único exemplo isolado. -> Também combate o overfitting.
        * **Por quê?** Este parâmetro olha para o *resultado* de uma divisão. Se uma pergunta fosse dividir um grupo de 30 amostras em dois novos grupos: um com 29 e outro com apenas 1, `min_samples_leaf=5` **proibiria** essa divisão. Ele força que qualquer regra criada resulte em grupos finais com um mínimo de 5 amostras cada. Isso impede que a árvore crie regras super específicas para isolar um único "dado teimoso", garantindo que cada previsão final seja baseada em um consenso de um grupo, e não em um ponto fora da curva.

* `random_state=42` (ou qualquer número)
    * **IMPORTANTÍSSIMO P/ REPRODUZIBILIDADE!**
    * Garante que o resultado seja **sempre o mesmo** toda vez que a gente rodar o código.
    * A árvore usa um pouco de aleatoriedade, então sem isso, cada execução pode dar um resultado um tiquinho diferente. Usar um número fixo garante que nosso experimento seja confiável.
        * **Por quê?** De onde vem a aleatoriedade? Se a árvore está em dúvida entre duas perguntas (ex: "idade > 40" e "salário > 50k") e ambas geram **exatamente a mesma melhoria** de pureza, o algoritmo precisa de um critério de desempate. Ele escolhe uma delas aleatoriamente. Se não usarmos o `random_state`, cada vez que rodarmos o código, ele pode desempatar de um jeito diferente, gerando uma árvore levemente diferente e, consequentemente, resultados diferentes. O `random_state=42` é como dizer: "toda vez que tiver um empate, desempate sempre do mesmo jeito".

