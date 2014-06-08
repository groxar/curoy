newiris <- iris
newiris$Species <- NULL
kc <- kmeans(newirirs, 3)
