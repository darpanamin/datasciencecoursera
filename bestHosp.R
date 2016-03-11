best <- function(state, outcome) {
  ## Read outcome data
  ## Check that state and outcome are valid
  ## Return hospital name in that state with lowest 30-day death
  ## rate
  outcomeValid <- outcome %in% colnames(hospDat)
  filepath <- file.path(getwd(),"R","outcome-of-care-measures.csv")
  hospDat <- read.csv(filepath, colClasses = "character",stringsAsFactors=FALSE,na.strings="Not Available")
  
  stateValid <- state %in% hospDat$State
  validOutcome <- c("heart attack", "heart failure", "pneumonia")
  heartAttack <- "Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack"
  heartFailure <- "Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure"
  pneumonia <- "Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia"
  
  if (stateValid == FALSE)
  {stop("invalid state")}
  
  
  if (outcome %in% validOutcome)
  {
    if (outcome == "heart attack") {outcomeCol <- heartAttack}
    if (outcome == "heart failure") {outcomeCol <- heartFailure}
    if (outcome == "pneumonia") {outcomeCol <- pneumonia}
  } else {stop("invalid outcome")}

  outcomeData <- hospDat[with(hospDat, State == state),c("Hospital.Name","City",outcomeCol) ]
  suppressWarnings(transform(outcomeData, outcomeCol = as.numeric(outcomeCol)))
  t <- na.omit(outcomeData)
  orderedRes <- t[order(as.numeric(t[,3]),t[,1]),]
  return(orderedRes[1,1])
}