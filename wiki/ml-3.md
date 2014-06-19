##ML3: Model Selection, Decision Making
###模型選擇
在曲線擬合的問題中，我們遇到了這樣一個問題，就是確定多項式的度(order of polynomial)。這個度決定了自由變量的個數，也決定了我們模型的複雜度。很多情況下，我們需要確定我們想要的模型的複雜度。而影響複雜度的因素可能會有很多，比如上面所說的多項式的度，再比如在regularized least squares中，$\lambda$的選擇。在實際的應用中，我們需要確定這樣的參數從而找到那個最具普遍性的模型。

對於maximum likelihood的方法來說，我們已經看到因爲存在overfitting的問題，我們不能通過模型在訓練集上的表現來選擇模型。如果數據夠多的話，一種方法是我們可以通過在一部分數據上訓練模型，而在另一部分數據上去檢測得到的模型，我們可以對數據進行不同的劃分，從而得到多個模型，並且進行選擇。這其中，用來進行檢測的數據叫做`validation set`。但是，即使是這樣，如果我們重複循環很多次的話，還是有overfitting的問題。這時，我們需要一個格外的數據集，`test set`，來對選擇出來的模型進行評估。
但是在一些時候，往往數據並不是那麼充分。這時一種解決方法是`cross validation`，我們拿所有數據的
$\frac{S-1}{S}$部分用於訓練，而拿剩餘的數據用於validation。當數據極爲缺乏時，我們可以把$S$設爲$N$，也就是數據點的個數，這就是`leave-one-out`。

cross validation的一個很大的缺點是計算量太大。這個是很顯然的。另外一個缺點是，如果模型的複雜度有很多的參數的話，那麼我們需要explore每一種參數的組合，這樣需要training的次數將會隨參數的個數指數增長。
因此我們需要尋找一種可以評估模型好壞並且只依賴訓練數據的方法。比如[`Akaike information criterion(AIC)`](http://en.wikipedia.org/wiki/Akaike_information_criterion)，還有[`Bayesian information criterion(BIC)`](http://en.wikipedia.org/wiki/Bayesian_information_criterion)。

###數據維數帶來的困難
對於模式識別問題，我們經常遇到的一個困難就是變量的維數`dimension`太高。以曲線擬合爲例，假如有$D$個變量，那麼一個度數爲3的多項式的一般式是
$$
y(x,w) = w\_0 + \sum\_{i=1}^{D} w\_i x\_i + \sum \_{i=1}^{D} \sum \_{j=1}^{D} w \_{ij} x\_i x\_j + \sum \_{i=1}^{D} \sum _{j=1}^{D} \sum     \_{k=1}^{D} w \_{ijk} x\_i x\_j x\_k
$$
這時我們知道，所需要估計的參數爲$2^D$。因此對與特別高維的變量，我們就無法在進行正常的曲線擬合了。高維變量帶來的問題被稱爲`curse of high dimensionality`。

雖然高維會給模式識別帶來很大的問題，但是我們仍然有一些辦法可以處理高維數據。主要原因有兩點：

1. 現實中的數據總是聚集在一個比較小的有效區域中。
2. 現實中的數據一般都是光滑(`smoothness`)的，因此輸入變量的小變化往往也只會導致輸出的微小變化，因此我們可以藉助於
類似`插值(interpolation)`的方法對新的輸出變量作出預測。

###決策理論
前面我們已經有了一個完整的框架來衡量事件的概率，那麼這裏我們要討論的是給定一些不確定時間的概率，我們應該怎麼樣作出最優的
決策。試想這樣一個情形，假如我們有病人的一張X光圖像，我們希望通過圖像來判斷病人是不是有癌症。在這樣一個例子中，輸入變量$x$
是圖像的特徵，而輸出則是一個分類，我們用$C\_k$來表示，$C_0$表示沒有癌症，而$C_1$表示有癌症。解決這樣一個問題包含兩個過程，
首先我們確定概率的分佈$p(x,C\_0)$和$p(x,C\_1)$, 這個過程叫做`inference`。第二個過程是根據得到的概率，決定病人是不是患有
癌症，這個過程稱爲`decision`。對於決策理論來說，不同的標準就有着不同的結論，下面我們簡單介紹幾個標準。

####決策標準
**Minimize misclassification rate** 我們需要一個規則把所有的sample分到一個現有的類中，這樣一個規則會把輸入劃分爲不同
的區域(decision regions)，我們用$R\_k$表示。不同區域之間的邊界被稱爲`決策邊界(decision boundaries, decision surfaces)`。
所有的區域都是互不相交的。那麼當一個屬於$C\_1$的被分到了$C\_2$或者相反，這時就出現了一個誤判，所以我們有如下定義，
$$
    p(mistake) = \int \_{R\_1} p(x,C\_2)dx + \int \_{R\_2} p(x,C\_1)dx
$$
很顯然，爲了使得$p(mistake)$最小化，假如$p(x,C_2) > p(x,C_1)$, 我們就應該把$x$分到$C_2$中，反之亦然。根據貝葉斯法則，我們知道
$p(x,C_2) = p(C_2 | x) * p(x)$，而$p(x)$是固定的。所以我們應該把$x$分到後驗概率$p(C_k|x)$最大的那個類中。

**Minimize the expected loss** 只是減少誤判的概率並不是一個很好的準測，因爲對於不同的誤判，帶來的後果嚴重程度是不同的。如果
一個沒有患病的人誤判爲患病，那麼最多只是接受了一些不必要的治療，而如果一個患病的人被誤判爲健康，那麼就可能帶來生命的損失。兩種誤判所帶來的後果
是完全不同的。爲了表示這種差異，我們引入`loss function`，或者稱爲`cost function`，我們的目標是最小化可能的損失。對於一個輸入$x$，假設它
屬於$C_k$但是我們把它分到了$C\_j$，這時我們引入一個loss，用$L_{kj}$表示。這樣我們就有了一個loss matrix L。那麼我們的loss function也就可以
定義爲
$$
    E(L) = \sum\_{k} \sum\_{j} \int\_{R\_j} L\_{kj} p(x,C\_k)dx
$$
同樣，裏面的$p(x,C_k)$ 可以用$p(C_k|x)$代替，因爲前驗概率是固定的。

####Reject option
對於不同的類$C_k$，當$p(C_k|x)$的值很接近是，我們就不能很好地確定輸入值到底屬於哪個類。爲了解決這樣一個情況，我們引入`reject option`。一個
常用的reject option是$max(p(C_k|x)) < \theta$。

<center>
<img src="https://dl.dropboxusercontent.com/u/47747425/Photo/Screen%20Shot%202014-06-19%20at%207.40.33%20PM.png" alt="reject option"/>
</center>

####Inference and decision
我們上面把分類問題分成了兩個階段，第一個階段是inference，第二個階段是decision。實際上我們有三種不同的方法來解決這樣的分類問題：

1. 首先計算$p(x|C_k)$，然後計算$p(C_k)$，由此我們可以計算出$p(x)$，之後根據貝葉斯我們可以得到$p(C_k|x)$。最後我們根據決策理論確定$x$所屬的類。這種方法
    被稱爲`generative model`。
2. 直接計算褚$p(C_k|x)$，然後根據決策理論確定$x$所屬的類。這種方法被稱爲`discriminative models`。
3. 計算一個函數$f$，將每一個輸入$x$映射到一個class label。比如說如果只有兩類，那麼$f=0$代表class 1，$f=2$代表class 2。這樣一個函數稱爲`discrimination function`。

我們可以分析一下三種方法的利弊。第一種方法要求最高，爲了計算$p(x|C_k)$和$p(C_k)$，我們可能需要一個很大的數據集。我們得到了所有有關概率的信息，但我們其實需要的僅僅是後驗概率。
然而第一個方法的好處是，這些額外的信息可以給我們提供一些便利。比如說我們可以計算$p(x)$從而發現在當前的模型下，數據點的概率分佈，對於一些低概率的數據點，我們對於他們的預測可能
準確性就會比較差，這被稱爲`outlier detection`，或者稱爲`novelty detection`。第二種方法直接計算後驗概率。第三種方法則是要找到下圖中綠線對應的$x$值，從而獲得最小的
misclassification rate。

<center>
<img src="https://dl.dropboxusercontent.com/u/47747425/Photo/Screenshot%202014-06-19%2022.02.12.png" />
</center>























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
