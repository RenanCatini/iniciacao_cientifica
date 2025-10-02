# Anotações sobre Modelos de Machine Learn


# Anotações - Árvore de Decisão 🌳

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


----

# Anotações - Support Vector Machine (SVM) 🦾

### Ideia Principal
* Pense em separar dois grupos de pontos em um papel.
* O SVM não quer só traçar uma linha qualquer pra separar. Ele quer encontrar a **"avenida" mais larga possível** entre os dois grupos.
* Os pontos que ficam na beirada dessa avenida são os mais importantes.

### O que é?
* Modelo de ML usado principalmente para classificação.
* Busca encontrar um "hiperplano" (uma linha, num plano 2D, ou um plano, num espaço 3D, etc.) que melhor divide os dados em classes.
* O "melhor" hiperplano é aquele com a **margem máxima**.

### Como funciona? (A parte importante)

**A grande dúvida:** Como o SVM decide qual é a "melhor" linha/avenida para separar os grupos?

**Resposta: MAXIMIZAR A MARGEM.**

* **Margem:** É a distância entre a linha de separação central e os pontos mais próximos de cada classe. Pense nela como a largura total da "avenida".
* **Vetores de Suporte (Support Vectors):** São os pontos de dados que ficam exatamente na beirada da margem (no "meio-fio" da avenida). Eles são os pontos mais difíceis de classificar e são os únicos que o modelo usa para definir a fronteira. Se a gente remover qualquer outro ponto que não seja um vetor de suporte, a "avenida" não muda.

**OBJETIVO DO ALGORITMO:** Encontrar a linha que cria a **avenida mais larga possível**, pois isso torna o modelo mais robusto para classificar novos dados.

* *E se os dados não puderem ser separados por uma linha reta? Aqui entra a mágica:*
* **O Truque do Kernel (Kernel Trick):** É uma função matemática que projeta os dados para uma dimensão maior, onde eles magicamente se tornam separáveis por uma linha (ou plano).
    * **Analogia:** Imagine pontos azuis no centro de um prato e vermelhos em volta (impossível separar com uma linha). O Kernel Trick seria como bater com força embaixo do prato, jogando os pontos azuis para o alto. Agora, em 3D, você pode passar uma "folha de papel" (um plano) horizontalmente para separar perfeitamente os pontos azuis dos vermelhos.

### Vantagens e Desvantagens

**👍 Vantagens:**
* Muito eficaz em espaços de alta dimensão (muitas features).
* Eficiente em termos de memória, pois usa apenas os "vetores de suporte".
* Versátil graças ao "Truque do Kernel".

**👎 Desvantagens:**
* **ESCALONAMENTO DE DADOS!!! (MUITO CUIDADO AQUI)**
* O que é? O SVM é muito sensível à escala das features. Se uma feature (ex: salário) tem valores na casa dos milhares e outra (ex: idade) na casa das dezenas, a feature de maior escala vai dominar o modelo.
* **Solução:** É **obrigatório** normalizar ou padronizar os dados (ex: com `StandardScaler` do `sklearn`).
* Pode ser lento em datasets muito grandes.

### Mão na Massa: O `SVC` do `sklearn` 🐍

Na prática, usamos a biblioteca `scikit-learn`. `SVC` significa "Support Vector Classification".

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. É CRUCIAL escalar os dados primeiro!
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
# X_teste_scaled = scaler.transform(X_teste) # Usa o mesmo scaler!

# 2. Criando o modelo com os parâmetros padrão
modelo_svm = SVC()

# 3. Depois é só treinar com os dados JÁ ESCALADOS
# modelo_svm.fit(X_treino_scaled, y_treino)
```

### Hiperparâmetros importantes para criar o modelo

```python
    SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
```

* `C=1.0` **(Parâmetro de Regularização)**
    * Controla o equilíbrio entre maximizar a margem e minimizar o erro de classificação.
    * **C baixo:** Prioriza uma **margem larga**, mesmo que isso signifique classificar errado alguns pontos do treino. É uma "margem suave" que gera um modelo mais simples e com **menor chance de overfitting**.
    * **C alto:** Tenta classificar **corretamente todos** os pontos de treino, o que pode levar a uma margem mais estreita e a um modelo mais complexo. **GRANDE CHANCE DE OVERFITTING!**
    > **Analogia:** `C` é o "preço" que o modelo paga por cada erro. Com `C` alto, o preço é caro, então ele evita erros a todo custo, se ajustando demais aos dados de treino.

* `kernel='rbf'` **(ou `'linear'`, `'poly'`)**
    * É aqui que a gente escolhe o "Truque do Kernel" para lidar com a complexidade dos dados.
    * `'linear'`: Para dados que você acredita serem linearmente separáveis. É o mais simples e rápido.
    * `'rbf'` (Radial Basis Function): É o padrão e um ótimo ponto de partida. Funciona bem para a maioria dos casos complexos e não-lineares, criando fronteiras baseadas em distância.
    * `'poly'`: Usa uma função polinomial para criar fronteiras curvas.

* `gamma='scale'` **(ou um número, ex: `0.1`)**
    * Define o alcance da influência de um único ponto de treino. **Só afeta kernels não-lineares como `'rbf'` e `'poly'`.**
    * **`gamma` baixo:** A influência de um ponto é **grande** (longo alcance). A fronteira de decisão é mais suave e geral. Pode levar a *underfitting*.
    * **`gamma` alto:** A influência de um ponto é **pequena** (curto alcance). A fronteira de decisão fica mais irregular e se ajusta muito aos dados de treino. Pode levar a *overfitting*.
    > **Por quê?** Um `gamma` alto significa que o modelo considera apenas os pontos muito próximos para tomar uma decisão, ignorando o "quadro geral". `gamma='scale'` (padrão) é uma escolha segura que se ajusta automaticamente com base nos seus dados.

* `random_state=42` **(ou qualquer número)**
    * **IMPORTANTÍSSIMO P/ REPRODUZIBILIDADE!**
    * Embora o algoritmo do SVM seja determinístico, ele usa um gerador de números aleatórios para algumas tarefas internas (como quando se usa `probability=True`).
    * Usar um `random_state` fixo garante que, sob as mesmas condições, o resultado será **sempre o mesmo**, o que é crucial para comparar experimentos e garantir a confiabilidade do seu trabalho.

