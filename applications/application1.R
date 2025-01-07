# Execute in terminal

# devtools::install_github("InseeFrLab/cartiflette-r", force = TRUE)
# mc cp s3/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet dvf.parquet
# mc cp s3/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet carreaux.parquet


library(duckdb)
library(cartiflette)
library(glue)
library(dplyr)
library(sf)

cog_malakoff <- "92046"
cog_montrouge <- "92049"

# Installer et charger les extensions spatiales
con <- dbConnect(duckdb::duckdb())
dbExecute(con, "INSTALL spatial;")
dbExecute(con, "LOAD spatial;")

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

malakoff <- communes92 |> filter(INSEE_COM == cog_malakoff)
montrouge <- communes92 |> filter(INSEE_COM == cog_montrouge)

# Zone à façon à l'intérieur de Malakoff
triangle <- sf::st_read("triangle.geojson")


# Partie 1

describe_dvf <- dbGetQuery(con, "DESCRIBE SELECT * FROM read_parquet('dvf.parquet')")
print(describe_dvf)

preview <- dbGetQuery(con, "SELECT * FROM read_parquet('dvf.parquet') LIMIT 10")
print(preview)

dbGetQuery(con, "SELECT DISTINCT nature_mutation FROM read_parquet('dvf.parquet')")
dbGetQuery(con, "
        SELECT
            MIN(valeur_fonciere) AS min_valeur,
            MAX(valeur_fonciere) AS max_valeur
        FROM read_parquet('dvf.parquet')
           ")

query1 <- glue("
    FROM read_parquet('dvf.parquet')
    SELECT
        code_commune,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY valeur_fonciere) AS mediane_valeur_fonciere
    WHERE code_commune IN ('{cog_malakoff}', '{cog_montrouge}')
    GROUP BY code_commune
")

result1 <- dbGetQuery(con, query1)
print(result1)


# Partie 2 ---------------------------------------------------------------------

query2 <- glue("
    FROM read_parquet('dvf.parquet')
    SELECT
        code_commune,
        valeur_fonciere,
        ST_AsText(geometry) AS geom_text
    WHERE code_commune = '{cog_malakoff}'
")

transactions_malakoff <- dbGetQuery(con, query2)

transactions_malakoff <- st_as_sf(transactions_malakoff, wkt = "geom_text") |>
  rename(geometry=geom_text) |>
  sf::st_transform(2154)

transactions_malakoff |> st_intersects(triangle)

query <- glue("
    FROM read_parquet('dvf.parquet')
    SELECT
      code_commune,
      valeur_fonciere,
      nature_mutation,
      ST_AsText(geometry) AS geom_text
    WHERE code_commune IN ('{cog_malakoff}', '{cog_montrouge}')
")

query <- glue("
    FROM read_parquet('dvf.parquet')
    SELECT
        code_commune,
        AVG(valeur_fonciere) AS moyenne_valeur_fonciere
    WHERE code_commune IN ('{cog_malakoff}', '{cog_montrouge}')
    GROUP BY code_commune
")
# 1. Définir une zone personnalisée à Paris
zone <- st_polygon(list(rbind(
  c(2.335, 48.85),   # Coin inférieur gauche
  c(2.355, 48.85),   # Coin inférieur droit
  c(2.355, 48.865),  # Coin supérieur droit
  c(2.335, 48.865),  # Coin supérieur gauche
  c(2.335, 48.85)    # Retour au premier point pour fermer le polygone
))) %>%
  st_sfc(crs = 4326)  # Définir le système de coordonnées (WGS84)

library(mapview)
zone_sf <- st_sf(geometry = zone)
mapview(zone_sf)


# Lire et interroger les données
carreaux_duckdb <- dbGetQuery(con, sprintf("
FROM read_parquet('carreaux.parquet')
SELECT
    *, ST_AsText(geometry) AS geom_text
WHERE st_distance(
        st_centroid(geometry),
        ST_Transform(st_point(%f, %f), 'EPSG:4326', 'EPSG:2154')
    ) / 1000 < 2
", reference_lat, reference_lon))

# Afficher le résultat
print(carreaux_duckdb)

# Déconnecter la connexion DuckDB
dbDisconnect(con)



#########################################


library(sf)       # Pour manipuler des données géographiques
library(duckdb)   # Pour interroger le fichier GeoParquet
library(dplyr)    # Pour la manipulation des données

# Créer une connexion DuckDB
con <- dbConnect(duckdb::duckdb())
dbExecute(con, "INSTALL spatial;")
dbExecute(con, "LOAD spatial;")



mapview(library(sf)       # Pour manipuler des données géographiques
library(duckdb)   # Pour interroger le fichier GeoParquet
library(dplyr)    # Pour la manipulation des données

# Convertir en texte WKT (Well-Known Text) pour DuckDB
zone_wkt <- st_as_text(zone)

# 2. Interroger les points dans le fichier GeoParquet avec DuckDB
geo_query <- sprintf(
  "
  FROM read_parquet('dvf.parquet')
  SELECT
    *,
    ST_AsText(geometry) AS geom_text
  WHERE
    ST_Within(
      geometry,
      ST_GeomFromText('%s')
    )
  ",
  zone_wkt
)

result <- dbGetQuery(con, geo_query)




geo_query <-
  "
  FROM read_parquet('dvf.parquet')
  SELECT
    *,
    ST_AsText(geometry) AS geom_text
  WHERE
    code_commune = '01243'
  "

geo_query <-
  "
  FROM read_parquet('dvf.parquet')
  SELECT
    *,
    ST_AsText(geometry) AS geom_text
  WHERE
    code_commune LIKE '92___'
  "


result <- dbGetQuery(con, geo_query)

result <- dbGetQuery(
  con,
  "
  FROM read_parquet('dvf.parquet')
  SELECT
  *,
  ST_AsText(geometry) AS geom_text
  "
)



# 3. Afficher les résultats
print(result)

# 4. Déconnecter DuckDB
dbDisconnect(con)
)
# Convertir en texte WKT (Well-Known Text) pour DuckDB
zone_wkt <- st_as_text(zone)

# 2. Interroger les points dans le fichier GeoParquet avec DuckDB
geo_query <- sprintf("
FROM read_parquet('carreaux_geoparquet.parquet')
SELECT
    *, ST_AsText(geometry) AS geom_text
WHERE ST_Within(
    geometry,
    ST_GeomFromText('%s')
)
", zone_wkt)

result <- dbGetQuery(con, geo_query)

# 3. Afficher les résultats
print(result)

# 4. Déconnecter DuckDB
dbDisconnect(con)

























