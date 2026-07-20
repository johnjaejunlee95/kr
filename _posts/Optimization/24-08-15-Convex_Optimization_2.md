---

layout: post
gh-repo: johnjaejunlee95/kr
gh-badge: [star,follow]
comments: true
author: johnjaejunlee95
title: "[개념설명] Convex Optimization 2"
date: "2024-08-18"
last-updated: "2024-08-18"
permalink: /Optimization_2/
description: ""
categories: [Optimization]
hits: true 
toc: true
tags: [Convex, Optimization, Gradient Descent]
use_math: true
author_profile: true
published: true
sidebar:
  nav: "docs"
---

<div>두번째 optimization 관련 post로 돌아왔습니다!! 이번에는 지난번에 function이 L-lipschitz일 때 어떻게 전개되는지, 어떻게 converge되는지 살펴보았는데요. 이번에는 좀 더 강한 constraint / assumption이 들어갈 때 어떻게 되는지 살펴보도록 하겠습니다. 그럼 거두절미하고 바로 들어가보도록 하겠습니다!! :smiley: </div>

### $\star$ Recap [(이전 Post)](https://johnjaejunlee95.github.io/kr/Optimization_1/)

이전 post에서 어떤 function이 <mark style="background: orange">convex</mark> + $L$-lipschitz이라고 가정했을 때 다음과 같은 Theorem이 나왔습니다:

<mark style="background:skyblue" >Theorem</mark> Let $f$ be convex and $L$-Lipschitz continuous. Then gradient descent with $\gamma = \frac{\mid\mid x\_1 - x^\star\mid\mid}{L\sqrt{T}}$  satisfies:

$$f \left ( \frac{1}{T} \sum_{k=1}^T x_k  \right) - f(x^{\star}) \leq \frac{\mid\mid x_1 - x^\star \mid\mid L}{\sqrt{T}} \Rightarrow \mathcal{O}(\frac{1}{\sqrt{T}})$$

그렇다면 만약 $L$-Lipschitz보다 더 강한 constraint 또는 assumption을 걸어주면 어떤 statement가 나올까요? 직관적으로 봤을 때  아마 훨씬 더 빠른 속도로 수렴할 것입니다. 그렇다면 실제로 더 빠른 속도로 수렴하는지 한번 증명을 통해 확인해보도록 하겠습니다. (모든 예시는 Gradient Descent 입니다!!)



### 들어가기에 앞서...

최적화 문제를 다룰 때 assumption/constraint를 다양하게 적용할 수 있습니다. (앞으로는 모든 가정/제한을 거는 상황에 대해서는 ***assumption***으로 표현하도록 하겠습니다.) Assumption이 강하면 강할수록 이를 만족하는 함수들이 적어지겠지만 convergence 속도는 빨라집니다. 따라서 이런 제약 조건들을 적절하게 배치를 하면 "왜 잘 동작하는지"에 대해 학습 이론(Learning Theory) 관점에서 확인할 수 있습니다. 그렇게 첫번째로 살펴보았던 assumption이 $L$-Lipschitz였고 이번에는 $\beta$-Smoothness 조건에서 어떻게 gradient descent가 수렴하는지 살펴보겠습니다!!

## Problem Formulation 2 <br> (Assumption: $\beta$ -Smooth)

기본적으로 $\beta$-Smoothness 의 정의는 다음과 같습니다:

<mark style="background:lightgreen" >Definition</mark> A continuously differentiable function $f$ is $\beta$-smooth if the gradient $\nabla f$ is $\beta$-Lipschitz:


$$
||  \nabla f(x) - \nabla f(y) ||  \leq \beta || x - y ||
$$


즉, 임의의 function $f$가 미분 가능한 형태라면 미분한 함수에 대해서 $\beta$-Lipschitz 하다는 의미입니다. 이게 왜 더 강한 assumption인지를 생각해보면 저희가 고등학교 때 배운 미분에 대한 개념을 생각해보시면 됩니다.

 $L$-Lipschitz의 경우는 임의의 function $f$ 에 대해서 변화율이 $L$에 비례해서 제한되는 것을 의미입니다. 그런데, $\beta$-Smooth는 임의의 function $f$의 미분값의 변화율이 $\beta$ 에 비례해서 제한되는 것을 의미합니다. 즉, $L$-Lipschitz는 function $f$의 변화율 $\rightarrow$ 미분값이 $L$에 비례한다는 것이고, $\beta$-Smoothness는 function $f$ 미분값의 변화율 $\rightarrow$ 2차 미분값이 $\beta$에 비례한다는 의미입니다. 이렇게 됐을 때, 모든 $\beta$-Smooth는 $L$-Lipschitz에 속하지만, 모든 $L$-Lipshcitz가 $\beta$-Smooth에 속하진 않기 때문에, **$\beta$-Smooth가 더 강한 assumption이라 볼 수 있습니다!!**

다시 본론으로 돌아와서 임의의 function $f$이 $\beta$-smooth일 때 gradient descent가 어떻게 converge 하는지 확인해보겠습니다!!

<mark style="background:skyblue" >Theorem 2:</mark> Let $f$ be convex and $\beta$-smooth on $\mathbb{R}^n$ \. Then, gradient descent with $\gamma = \frac{1}{\beta}$ satisfies


$$
f(x_T) - f(x^\star) \leq \frac{2\beta || x_1 - x^\star || ^2}{T-1}
$$


전개 방법은 이전에 $L$-Lipschitz 일때와 비슷합니다. 다만 좀 더 강한 assumption이 들어갔기 때문에 몇 가지만 더 살펴보면 됩니다

### Proof:

들어가기에 앞서 $\beta$-smooth의 성질로 인해 정의할 수 있는 definition이 하나 더 있습니다:


$$
f(y)  \leq f(x) + \left< \nabla f(x) , y-x\right> + \frac{\beta}{2} || y - x ||^2
$$


이 수식은 function $f$가 twice-differentiable하다는 assumption이 이미 있기 때문에, 이를 Talyor Expansion의 개념을 활용해서 전개를 할 수 있습니다. (이는 main point는 아니고 definition으로써 활용을 하는 것이므로 proof는 [링크](https://angms.science/doc/CVX/CVX_betasmoothsandwich.pdf)로 남겨두겠습니다!!)

그러면 이제 진짜로 $\beta$-Smoothness일 때의 gradient descent의 convergence에 대해 proof를 진행해보겠습니다. 전개 방향성 자체는 $L$-Lipschitz일때와 비슷합니다. 다만 추가로 활용할 수 있는 definition이 늘어났다고 보시면 됩니다. 우선은 $\beta$-Smooth의 성질에 의한 gradient descent 식을 다음과 같이 전개를 할 수 있습니다.


$$
\begin{align}
  f(x_{t+1}) - f(x_t) &\leq \left< \nabla f(x_{t}) , x_{t+1} - x_t \right> + \frac{\beta}{2} || x_{t+1} - x_t||^2 \\
  &= \left< \nabla f(x_t) , -\gamma\nabla f(x_t) \right> + \frac{\beta}{2} ||-\gamma\nabla f(x_t)||^2 \\
  &= -\gamma ||\nabla f(x_t)||^2 + \frac{\beta}{2}*\gamma^2 ||\nabla f(x_t)||^2\\
  &= -\frac{1}{\beta} ||\nabla f(x_t)||^2 + \frac{1}{2\beta}||\nabla f(x_t)||^2 \;\; \text{where } \gamma = \frac{1}{\beta}\\
  &= -\frac{1}{2\beta}||\nabla f(x_t)||^2
  
\end{align}
$$



또한 $L$-Lipschitz 일때의 proof처럼 $f(x_t) - f(x^\star)$ 의 bound를 $f$의 convexity를 활용하면 다음과 같이 전개할 수 있습니다: $f(x_t) - f(x^\star) \leq \left< \nabla f(x_t) , x_t - x^\star \right>$

여기서 우리가 활용할 수 있는 term은 위의 proof를 생각했을 때 $\nabla f(x_t)$ 입니다.  즉, $\nabla f(x_t)$의 제곱형태인 $\mid\mid \nabla  f(x_{t})\mid\mid^2 $ 와 관련된 bound가 이미 나와 있으므로 이를 활용하면 조금 더 원활히 식을 전개할 수 있습니다.  이는 **Cauchy-Schwarz Inequality** 성질을 활용해서 진행할 수 있습니다. 

 ($\star$ **Cauchy-Schwarz Inequality**: $\left< a,b \right> \leq \mid\mid a \mid\mid \cdot \mid\mid b \mid\mid $)


$$
\begin{align}
\left[f(x_t) - f(x^\star)\right]^2 &\leq \left| \left< \nabla f(x_t) , x_t - x^\star \right>\right|^2 \\
&\leq || \nabla f(x_t) ||^2 \cdot || x_t - x^\star ||^2 \\
\Rightarrow \frac{\left[f(x_t) - f(x^\star)\right]^2}{ || x_t - x^\star ||^2} &\leq || \nabla f(x_t) ||^2 
\end{align}
$$


자, 그러면 다음 단계로는 우리가 다루기 쉬운 형태로 re-arrange를 해야하는데, 제일 쉬운 방법은 결국 $f(x_{t+1}) - f(x_t)$ 에 대한 bound 를 다루는 것입니다. 이는 다음과 같이 양쪽 inequality term에 $f(x^\star)$를 빼주면 이전에 전개했던 방법들을 모두 활용할 수 있게 됩니다. 

다만, 그 전에 한가지 체크할 부분이 있습니다. $f$에 대해서 살펴보기 전에 step $t$에 따라 bound가 어떻게 생기는지 확인해야 합니다. 따라서, $f(x_{t+1}) - f(x_t)$ 를 살펴보기 전에  $x_t$와 $x_{t+1}$간의 bound가 어떻게 형성이 되는지 확인해보겠습니다.


$$
\begin{align}
	|| x_{t+1} - x^\star ||^2 &= || x_t - \frac{1}{\beta} \nabla f(x_t) - x^\star ||^2 \\
	&= ||x_t -x^\star||^2 - \frac{2}{\beta} \underbrace{\left< \nabla f(x_t), x_t - x^\star \right>}_{\leq \frac{1}{\beta} ||\nabla f(x_t)||^2} +\frac{1}{\beta^2} ||\nabla f(x_t)||^2 \\
	&\leq ||x_t - x^\star||^2 \underbrace{- \frac{1}{\beta^2} ||\nabla f(x_t)||^2}_{\text{always decreases}} \\
	&\leq ||x_t - x^\star||^2 \underbrace{- \frac{1}{\beta^2}\frac{\left[f(x_t) - f(x^\star)\right]^2}{ || x_t - x^\star ||^2}}_{\text{smaller decreases}}
\end{align}
$$

이러한 성질을 활용하여 $f(x^\star)$를 활용한 proof를 계속해서 다음과 같이 진행할 수 있습니다. 


$$
\begin{align}
	f(x_{t+1}) &\leq f(x_t) - \frac{1}{2\beta} ||\nabla f(x_t)||^2 \\
	f(x_{t+1}) - f(x^\star) = &\leq f(x_t) - f(x^\star) - \frac{1}{2\beta}||\nabla f(x_t)||^2 \\
	D_{t+1} &\leq D_t  - \frac{1}{2\beta}||\nabla f(x_t)||^2 \quad \text{where } D_{t+1} = f(x_{t+1}) - f(x^\star)\\
	D_{t+1} &\leq D_t - \frac{1}{2\beta}\cdot \frac{\left[f(x_t) - f(x^\star)\right]^2}{||x_t - x^\star ||^2} \\
	D_{t+1} &\leq D_t - \frac{1}{2\beta} \cdot \frac{D_t^2}{||x_t - x^\star ||^2}
\end{align}
$$



위 inequalities에 $D_t D_{T+1}$로 나눠주면 


$$
\begin{align}
	\frac{1}{D_{t}} - \frac{1}{D_{t+1}} &\leq - \frac{1}{2\beta||x_t - x^\star ||^2}\frac{D_t}{  D_{t+1}}
\end{align}
$$


가 됩니다. 그런데 여기서 2가지 조건들을 통해 bound를 재설정할 수 있습니다.

1. ${D_t}/{D_{t+1}}  = \left[f(x_t) - f(x^\star)\right] / \left[f(x_{t+1}) - f(x^\star)\right]$  이고 $f(x_{t+1})$ 이 $f(x_t)$ 보다 작기 때문에 ${D_t}/{D_{t+1}} \geq 1$ 인 것을 알 수 있음
2. ${x_t - x^\star} \leq x_1 - x^\star $  이므로 $1 /{\mid\mid x_t - x^\star \mid\mid^2} \geq 1 / {\mid\mid x_1 - x^\star \mid\mid^2}$ 임을 알 수 있음

그리고 Right Hand Side (RHS) term이 negative form 이므로 좀 더 tight하게 bound를 잡으면 다음과 같이 표현할 수 있습니다.


$$
\frac{1}{D_{t+1}} - \frac{1}{D_{t}} \geq \frac{1}{2\beta||x_1 - x^\star ||^2}
$$


지난번 전개와 마찬가지로 순차적으로 $t=1, \ldots , {T-1}$를 대입을 해보면


$$
\begin{align}
\frac{1}{D_{2}} - \frac{1}{D_{1}} &\geq \frac{1}{2\beta||x_1 - x^\star ||^2} \\
\frac{1}{D_{3}} - \frac{1}{D_{2}} &\geq \frac{1}{2\beta||x_1 - x^\star ||^2} \\
	&\;\;\vdots \\
\frac{1}{D_{T}} - \frac{1}{D_{T-1}} &\geq \frac{1}{2\beta||x_1 - x^\star ||^2} \\
\end{align}
$$


그리고 위 inequalites에 대해 summation을 진행하면



$$
\begin{align}
	\sum_{t=1}^{T-1} \left[\frac{1}{D_{t+1}} - \frac{1}{D_{t}} \right] &\geq \sum_{t=1}^{T-1} \left[\frac{1}{2\beta ||x_1 - x^\star ||^2}\right] \\
	\frac{1}{D_T} - \frac{1}{D_1} &\geq \frac{T-1}{2\beta ||x_1 - x^\star ||^2} \\
	\frac{1}{D_T} &\geq \frac{1}{D_1} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2}
	\end{align}
$$


여기서 $\beta$-Smooth의 property 와 convexity의 properties를 활용하여 $D_1$을 다음과 같이 설정할 수 있습니다.



$$
\begin{align}
	D_1 = f(x_1) - f(x^\star) &\leq \left< \nabla f(x^\star), x_1 - x^\star \right> + \frac{\beta}{2} ||x_1 - x^\star ||^2\\
	&\leq \frac{\beta}{2} ||x_1 - x^\star ||^2

\end{align}
$$

이게 가능한 이유는 결국 $x^\star$는 minimizer 인 값이기 때문에 $f$에 대한 미분값은 0으로 수렴한다고 가정할 수 있습니다. 따라서 $\nabla f(x^\star) = 0$ 이 되므로 위와 같은 식을 도출할 수 있습니다.



그리고 수식을 정리하면


$$
\begin{align}
	\frac{1}{D_T} &\geq \frac{1}{D_1} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2}\\
	\frac{1}{f(x_T) - f(x^\star)} &\geq \frac{1}{f(x_1) - f(x^\star)} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2} \\
	\frac{1}{f(x_T) - f(x^\star)} &\geq \frac{2}{\beta ||x_1 - x^\star ||^2} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2} \\
	\frac{1}{f(x_T) - f(x^\star)} &\geq \frac{T + 3}{2\beta||x_1 - x^\star ||^2}\\
	&\geq \frac{T -1}{2\beta||x_1 - x^\star ||^2}
\end{align}
$$


위에서 분자를 $T-3 \Rightarrow T-1$ 바꿔준 이유는, 결국 우리가 summation 을 했을 때 $1$부터 $T-1$ step까지 확인을 했기 때문에 꼴을 맞춰주는게 bound를 설정하기 편하기 때문입니다. 

이제 찐막으로 각 term들에 대해서 역수를 취해주면 끝납니다!! **(inequality 방향 바뀜)**


$$
\begin{align}
	f(x_T) - f(x^\star) \leq \frac{2\beta || x_1 - x^\star || ^2}{T-1} \Rightarrow \mathcal{O} \left(\frac{\beta}{T}\right)
	\end{align}
$$


## 이 후...

수식을 전개함에 있어서 최대한 설명을 잘 하려 했는데  만족시킬 수 있는 조건들이 늘어나다보니 설명이 조금 장황해진 것 같습니다. 때문에 가독성 및 전개가 매끄럽지 못하다는 느낌을 받으실 수도 있을 것 같네요. (죄송합니다...ㅠㅠ) 시간 날때마다 조금씩 다듬어보도록 하겠습니다.

Convex 관련한 증명은 여기까지 하겠습니다. (대부분 비슷하게 증명됩니다.) 다음 post에는 다른 컨텐츠로 돌아오겠습니다. :smiley:



<br>

**$\*\$읽어주셔서 매우 감사합니다!! 혹시나 글을 읽으시다가 틀린 부분이 있거나 조언해주실 부분이 있다면 언제든 의견 전달주시면 감사하겠습니다!!** :smiley:
