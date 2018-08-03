install.packages('spatstat')
library('spatstat')
install.packages('rmutil')
library('rmutil')

N=10

pp <- rpoispp(function(x,y) {73 * exp(-3*x-2*y)}, 100, nsim=N)
f = function(x,y) {73 * exp(-3*x-2*y)}


n.array = numeric()
for(i in 1:N){
  n.array[i] = pp[[i]]$n
  print(pp[[i]]$n)
  plot(pp[[i]], main=i)
}

a = max(n.array)

res.x = matrix(0, nrow=a, ncol=N)
res.y = matrix(0, nrow=a, ncol=N)
for(i in 1:N){
  res.x[1:pp[[i]]$n, i]= pp[[i]]$x
  res.y[1:pp[[i]]$n, i]= pp[[i]]$y
}

res.x
res.y

install.packages('reshape')
library(reshape)

dt1 <- melt(res.x)
dt1 <- dt1[,-1]
dt1
