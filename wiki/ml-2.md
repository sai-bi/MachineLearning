##ML2: Probability Theory

模式識別中一個很重要的概念就是**不確定性**(uncertainty)。不確定性既可以能來自於數據中的噪音(noise)也可能來自于有限的數據集。概率論為我們提供了一個完備的框架來對不確定性進行量化和操作。

###基礎的法則
加法法則
$$p(X) = \sum_{Y}p(X,Y)$$
乘法法則
$$p(X,Y) = p(Y|X)p(X)$$
貝葉斯定理
$$p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}$$

###先驗概率，后驗概率
$p$的先驗概率(prior),是在觀測到一些證據之前或者說這些證據被納入考慮之前$p$的概率分佈。而其后驗概率(posterior probability)則是在考慮其相關證據或數據后得到的概率分佈。

###概率密度
我們用概率密度(probability density)來描述連續變量的概率分佈。假設實數$x$落在區間
$[x,x+\delta x]$的概率由$p(x)\delta x, \delta x \rightarrow 0$給出,
那麼$p(x)$ 被稱為$x$的概率密度函數。由此，我們知道$x$落在區間$(a,b)$的概率是
$$ p(x \in (a,b)) = \int_{a}^{b} p(x)dx$$

###期望，方差
函數$f(x)$的期望(expectation)是$f(x)$在概率分佈$p(x)$下的均值(average value)。
$$ E(f) = \int p(x)f(x)dx $$
如果我們從$x$的概率分佈中任意抽出$N$個點，那麼$f(x)$的期望可以近似用這些點的均值來表示：
$$E(f) \approx \frac{1}{N} \sum_{n=1}^{N} f(x_n)$$
這一性質在我們討論**抽樣方法**(sampling method)的時候將會經常用到。同樣我們也會經常用到條件期望(conditional expectation):

$$ E_x[f|y] = \sum_{x}p(x|y) f(x)$$ 

$f(x)$ 的方差被定義為
$$ var[f] = E[(f(x)-E(f(x))^2] 
= E[f(x)^2] - E[f(x)]^2 $$
方差衡量了$f(x)$ 偏離其均值的多少。對於兩個變量$x,y$,他們的協方差(covariance)定義為
$$ cov[x,y] = E_{x,y} [x,y] - E(x)E(y)$$
協方差衡量了$x$和$y$協同變化的程度。

###貝葉斯概率
關於概率論，有兩種理解，一種基於隨機重複事件，被稱為經典概率論(classical interpretation of probability or frequentist probability)，而另外一種則把
概率用於衡量事件的不確定性，這種被稱為貝葉斯概率論。比如說我們考慮北極冰帽在本世紀內消失的概率，對於這樣的事件，不可能通過頻率
來定義，但是我們卻可以通過一些相關的信息來得到一些基本的觀點，比如說溫室氣體的排放量，還是衛星的數據，這些證據會讓我們能夠修正
事件的不確定性。We would like to be able to quantify our expression of uncertainty and make precise revisions 
of uncertainty in the light of new evidence, as well as subsequently to be able to take optimal actions or decisions
as a consequence。這些都可以通過貝葉斯概率實現。

以曲線擬合為例，$p(\mathbf{w})$給出了在我們看到訓練數據$D$之前參數$\mathbf{w}$的概率。而在看到數據集
$D={t_1,t_2,...,t_N}$之後，根據貝葉斯定理，我們有
$$ p(w|D) = \frac{p(D|w)p(w)}{p(D)}$$
數據集對概率的影響通過$p(D|w)$來表現，它是關於$w$的一個函數，被稱為似然函數(likelihood function)，似然函數
表示數據集$D$在不同的曲線參數$w$條件下出現的概率。而$p(w)$則是先驗概率。由此我們知道$posterior \propto
likelihood \times prior $。似然函數$p(D|w)$無論是在經典概率論還是在貝葉斯概率論中都起著至關重要的作用，然而形式卻
不盡相同。在經典概率論中，$w$是確定的(通常通過某種estimator確定)參數，而這一參數的error estimate通過數據集$D$的概率
分佈得到。而在貝葉斯概率論中，只有一個數據集$D$，參數$w$的不確定性是通過其概率分佈表示。

在經典概率論中，最常用的用來確定$w$取值的estimator是*maximum likelihood*：$w$取使似然函數取得最大值的那個值，意思是
說選擇使得我們觀察到數據集$D$概率最大的$w$值。同時，誤差函數(error fucntion)被定義為$-log p(D|w)$。最大化似然函數
等同於最小化誤差函數。那麼怎麼確定我們得到的$w$的誤差呢？一個經常採用的方法是*bootstrap*，具體的操作如下:

1. 給定數據集$X = {x_1,x_2,...,x_N}，我們有放回地從中隨機抽出$N$個點，組成新的數據集$X_B$
2. 按照1中的方法，生成多個這樣的數據集
3. 在每一個新生成的數據集上，按照最大似然函數估計新的$w$值
4. 那麼$w$的準確性可以通過其與新生成的$w$之間的差異(variability)來估計

對於貝葉斯概率論最常見的批評是，先驗概率的選擇經常是為了數學上的方便而不能真實地反映事實。

###高斯分佈
高斯分佈是最常用的一種分佈，它告訴了我們觀察數據落在某一區間的概率。對於單變量$x$, 它的高斯分佈被定義為
$$ N(x|\mu,\sigma ^2) = \frac{1}{\sqrt{(2 \pi \sigma ^2 )}} exp\\{-\frac{1}{2\sigma ^2}(x-\mu)^2\\}$$
其中$u$是$x$的平均值，$\sigma ^2 $是方差，$\sigma$被成為標準差，方差的倒數$\beta = 1/{\sigma ^2}$
被稱為*精確度*(precision)。
我們可以證明高斯分佈的確是正態的(normalized)，也就是我們可以證明高斯分佈的積分等於1([Wikipedia](http://en.wikipedia.org/wiki/Gaussian_integral))。
對於多元變量，我們同樣可以定義它的高斯分佈。

假設我們現在有一個數據集$x = [x_1,x_2,...,x_N]$,這些點是從一個$\mu$和$\sigma$未知的高斯分佈中獨立
(independent and identically distributed)取出，那麼
我們怎麼估計這個高斯分佈的參數呢？這裡我們採用最大似然的方法來估計未知的參數。因為所有的點都是互相獨立的，因此，我們有
$$p(x|\mu,\sigma ^2) = \Pi_{i=1}^{N} N(x_n|\mu,\sigma ^2)$$

這裡我們要讓這個概率最大。為了簡單方便，我們讓它相對應的$log$值最大，也就是最大化$ln(p)$。對於
$\mu$,我們令$\frac{\partial p}{\partial \mu} = 0$, 我們可以得到maximum likelihood solution：
$ \mu'  = \frac{1}{N} \sum _{i=1} ^{N} x_i $,
也就是sample mean。

同樣對於variance,我們可以利用同樣的方法使得偏導數 
$ \frac{\partial p}{\partial (\sigma ^2)} = 0$，等到相對應的solution，
$ {\sigma ^2}' = \frac{1}{N} \sum_{i=1}^{N} (x_i - u')^2 $，也就是sample variance。

###Maximum Likelihood的侷限性
上面我們通過maximum likelihood得到了估計的均值與方差，但實際上，這些結果是存在系統偏差的(systemic bias)。對於通過
maximum likelihood得到的$\mu',{\sigma^2}'$,以及真實的$\mu, \sigma^2$,他們之間存在這這樣的關係
$$
\begin{align}
E(\mu) & = \mu ' \\\
E({\sigma ^2}') &= \frac{N-1}{N} \sigma ^2
\end{align}
$$
這其中的道理也很簡單，根據期望的性質，我們有：
$$
E(\mu ') = \frac{1}{N} \sum_{i=1}^{N} 
E(x \_i) = \mu
$$
對於方差，稍微麻煩了一點。

To be continued...
  
`
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ TeX: { extensions: ["autobold.js"] }});
</script>
<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>
<script type="text/javascript"
src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>






























