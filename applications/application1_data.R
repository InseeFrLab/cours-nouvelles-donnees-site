# devtools::install_github("InseeFrLab/cartiflette-r", force = TRUE)
library(dplyr)

# Création des contours de Malakoff et Montrouge
communes92 <- cartiflette::carti_download(
  values = c("92"),
  borders = "COMMUNE",
  vectorfile_format = "geojson",
  filter_by = "DEPARTEMENT",
  year = 2022,
  provider = "IGN",
  source = "EXPRESS-COG-CARTO-TERRITOIRE",
  crs = 4326
)

cog_malakoff <- "92046"
cog_montrouge <- "92049"

malakoff <- communes92 |> filter(INSEE_COM == cog_malakoff)
montrouge <- communes92 |> filter(INSEE_COM == cog_montrouge)


BUCKET_OUT = "projet-formation"
FILE_MALAKOFF = "nouvelles-sources/data/malakoff.geojson"

aws.s3::s3write_using(
  malakoff,
  FUN = sf::st_write,
  object = FILE_MALAKOFF,
  bucket = BUCKET_OUT,
  opts = list("region" = "")
)

FILE_MONTROUGE = "nouvelles-sources/data/montrouge.geojson"

aws.s3::s3write_using(
  montrouge,
  FUN = sf::st_write,
  object = FILE_MONTROUGE,
  bucket = BUCKET_OUT,
  opts = list("region" = "")
)

