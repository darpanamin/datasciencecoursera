filePath <-"./Rstuff/getdata-projectfiles-UCI HAR Dataset/UCI HAR Dataset/"

features<-read.delim(paste0(filePath,"features.txt"), header = FALSE,stringsAsFactors = FALSE, sep=" ")

ftest<-paste0(filePath,"test/")
ftrain<-paste0(filePath,"train/")

file_list <- sapply(append(ftest,ftrain),list.files,pattern="*.txt",full.names = TRUE)
file_names <- sapply(append(ftest,ftrain),list.files,pattern="*.txt")
myfiles = lapply(file_list,fread)
names(myfiles) <-file_names

total_data <-rbind(cbind(myfiles$subject_test.txt,myfiles$y_test.txt,myfiles$X_test.txt),cbind(myfiles$subject_train.txt,myfiles$y_train.txt,myfiles$X_train.txt))
col_names<-append(c("subject","activity"),features$V2)
names(total_data)<-col_names