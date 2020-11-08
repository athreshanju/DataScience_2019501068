setwd("C:\\Users\\Chowdary\\Desktop\\Final exam\\Final exam")
bse_sensex <- read.csv("BSE_Sensex_Index.csv")
SGR_Close <- c()
for (i in 1:15446){
  SGR_Close[i] <- (bse_sensex$Close[i] - bse_sensex$Close[i+1]) / bse_sensex$Close[i+1]
}
SGR_Close[15447] <- (SGR_Close[15446] + SGR_Close[15445] + SGR_Close[15444]) / 3
SGR_Close[15447]
Z_SGR_Close <- c()
mean_SGR_Close <- mean(SGR_Close)
mean_SGR_Close
sd_SGR_Close <- sd(SGR_Close)
sd_SGR_Close
for (j in 1:15447) {
  Z_SGR_Close[j] <- (SGR_Close[j] - mean_SGR_Close) / (sd_SGR_Close) 
} 
outliers_dates <- c()
outliers_count <- 0
otd <- 1
for (k in 1:15447) {
  if (Z_SGR_Close[k] > 3) {
    outliers_count <- outliers_count + 1
    outliers_dates[otd] <- bse_sensex$Date[k]
    otd <- otd + 1
  }
  if (Z_SGR_Close[k] < -3) {
    outliers_count <- outliers_count + 1
    outliers_dates[otd] <- bse_sensex$Date[k]
    otd <- otd + 1
  }
}
outliers_count
outliers_dates