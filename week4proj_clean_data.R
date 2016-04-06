filePath <-"./Rstuff/getdata-projectfiles-UCI HAR Dataset/UCI HAR Dataset/"

features<-read.delim(paste0(filePath,"features.txt"), header = FALSE,stringsAsFactors = FALSE, sep=" ")
activity<-read.delim(paste0(filePath,"activity_labels.txt"), header = FALSE,stringsAsFactors = FALSE, sep=" ")
ftest<-paste0(filePath,"test/")
ftrain<-paste0(filePath,"train/")

file_list <- sapply(append(ftest,ftrain),list.files,pattern="*.txt",full.names = TRUE)
file_names <- sapply(append(ftest,ftrain),list.files,pattern="*.txt")
myfiles = lapply(file_list,fread)
names(myfiles) <-file_names
activity_data_test <-merge(myfiles$y_test.txt,activity, by.x = "V1", by.y = "V1", all.x = TRUE)
total_data <-rbind(cbind(myfiles$subject_test.txt,merge(myfiles$y_test.txt,activity, by.x = "V1", by.y = "V1", all.x = TRUE),myfiles$X_test.txt),cbind(myfiles$subject_train.txt,merge(myfiles$y_train.txt,activity, by.x = "V1", by.y = "V1", all.x = TRUE),myfiles$X_train.txt))
col_names<-append(c("subject","activity_id","activity_name"),features$V2)
names(total_data)<-col_names