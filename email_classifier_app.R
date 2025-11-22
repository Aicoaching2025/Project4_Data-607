library(shiny)
library(tidyverse)
library(tm)
library(e1071)

# Load the  trained model

nb_model <- readRDS("nb_model.rds")
top_terms <- readRDS("top_terms.rds")
train_features <- readRDS("train_features.rds")

# Prediction function
predict_email <- function(subject, body_text) {
  full_text <- paste(subject, body_text, sep = " ")
  
  clean_text <- full_text %>%
    str_to_lower() %>%
    str_replace_all("<[^>]+>", " ") %>%
    str_replace_all("http\\S+|www\\S+", "") %>%
    str_replace_all("\\S+@\\S+", "") %>%
    str_replace_all("[^a-z\\s]", " ") %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
  
  test_corpus <- VCorpus(VectorSource(clean_text))
  test_dtm <- DocumentTermMatrix(
    test_corpus,
    control = list(
      weighting = weightTfIdf,
      removePunctuation = TRUE,
      removeNumbers = TRUE,
      stopwords = TRUE,
      stemming = FALSE,
      dictionary = top_terms
    )
  )
  
  test_feat <- as.data.frame(as.matrix(test_dtm))
  
  missing_cols <- setdiff(colnames(train_features)[-ncol(train_features)], colnames(test_feat))
  for(col in missing_cols) {
    test_feat[[col]] <- 0
  }
  
  test_feat <- test_feat[, colnames(train_features)[-ncol(train_features)]]
  
  prediction <- predict(nb_model, test_feat)
  return(as.character(prediction))
}

# UI
ui <- fluidPage(
  titlePanel("📧 Gmail Email Classifier"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Enter Email Details"),
      
      textInput("subject", 
                "Subject Line:", 
                placeholder = "e.g., 50% Off Sale Today!"),
      
      textAreaInput("body", 
                    "Email Body:", 
                    placeholder = "Paste your email text here...",
                    rows = 10),
      
      hr(),
      
      h4("OR Upload Text File"),
      fileInput("file", 
                "Choose .txt file",
                accept = c("text/plain", ".txt")),
      
      hr(),
      
      actionButton("predict", 
                   "Classify Email", 
                   class = "btn-primary",
                   style = "width: 100%;")
    ),
    
    mainPanel(
      h3("Prediction Results"),
      
      wellPanel(
        h2(textOutput("prediction"), style = "color: #2c3e50; text-align: center;"),
        br(),
        htmlOutput("category_info")
      ),
      
      hr(),
      
      h4("About This Classifier"),
      p("This email classifier uses a Naive Bayes model trained on 3,200 emails 
        across 4 categories:"),
      tags$ul(
        tags$li(tags$b("Inbox:"), " Personal and important emails"),
        tags$li(tags$b("Promotions:"), " Marketing and sales emails"),
        tags$li(tags$b("Social:"), " Social media notifications"),
        tags$li(tags$b("Updates:"), " Order confirmations, shipping updates")
      ),
      p("The model uses TF-IDF features and achieved high accuracy on test data."),
      
      hr(),
      p(em("Created by Candace for DATA 643 - Recommender Systems"), 
        style = "text-align: center; color: #7f8c8d;")
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Get email text (either from inputs or file)
  get_email_text <- reactive({
    
    # If file is uploaded, use that
    if (!is.null(input$file)) {
      file_content <- readLines(input$file$datapath, warn = FALSE)
      body_text <- paste(file_content, collapse = " ")
      
      # Try to extract subject from first line if it starts with "Subject:"
      if (grepl("^subject:", tolower(file_content[1]))) {
        subject <- sub("^subject:\\s*", "", file_content[1], ignore.case = TRUE)
        body_text <- paste(file_content[-1], collapse = " ")
      } else {
        subject <- "Email from uploaded file"
      }
      
      return(list(subject = subject, body = body_text))
    } else {
      # Use text inputs
      return(list(subject = input$subject, body = input$body))
    }
  })
  
  # Make prediction when button is clicked
  prediction_result <- eventReactive(input$predict, {
    email <- get_email_text()
    
    # Validate inputs
    if (email$subject == "" && email$body == "") {
      return(list(category = "ERROR", 
                  message = "Please enter email text or upload a file!"))
    }
    
    # Make prediction
    category <- predict_email(email$subject, email$body)
    
    return(list(category = category, message = NULL))
  })
  
  # Display prediction
  output$prediction <- renderText({
    result <- prediction_result()
    
    if (result$category == "ERROR") {
      "⚠️ No Input Provided"
    } else {
      paste0("📁 Category: ", toupper(result$category))
    }
  })
  
  # Display category information
  output$category_info <- renderUI({
    result <- prediction_result()
    
    if (result$category == "ERROR") {
      HTML("<p style='color: #e74c3c;'>Please enter an email subject and body, 
           or upload a text file to classify.</p>")
    } else {
      
      info <- switch(result$category,
        "inbox" = list(
          icon = "✉️",
          color = "#3498db",
          desc = "This appears to be a personal or important email that belongs in your main inbox."
        ),
        "promotions" = list(
          icon = "🏷️",
          color = "#e67e22",
          desc = "This looks like a promotional or marketing email with offers, sales, or advertisements."
        ),
        "social" = list(
          icon = "👥",
          color = "#9b59b6",
          desc = "This appears to be a social media notification about likes, comments, or connections."
        ),
        "updates" = list(
          icon = "📦",
          color = "#27ae60",
          desc = "This looks like a transactional email such as order confirmations, shipping updates, or receipts."
        )
      )
      
      HTML(paste0(
        "<div style='background-color: ", info$color, "20; padding: 15px; border-left: 4px solid ", info$color, ";'>",
        "<p style='font-size: 18px;'>", info$icon, " <b>", toupper(result$category), "</b></p>",
        "<p>", info$desc, "</p>",
        "</div>"
      ))
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)
