exams<-read.csv("spring2008exams.csv")
str(exams)
q1=quantile(exams$Midterm.2,.25,na.rm=TRUE)
q2=quantile(exams$Midterm.2,.75,na.rm=TRUE)
qr<-q2-q1
qr
exams[(exams$Midterm.2>q2+1.5*qr),3]
exams[(exams$Midterm.2>q1-1.5*qr),3]
boxplot(exams$Midterm.1,exams$Midterm.2,col="green",main="Exam Scores",names=c("Exam 1","Exam 2"),ylab="Exam Score")