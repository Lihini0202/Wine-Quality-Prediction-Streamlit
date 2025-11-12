terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "wine_rg" {
  name     = "wine-predictor-rg"
  location = var.region
}

resource "azurerm_container_registry" "wine_acr" {
  name                = "winepredictoracr${random_id.acr_suffix.hex}"
  resource_group_name = azurerm_resource_group.wine_rg.name
  location            = azurerm_resource_group.wine_rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

resource "random_id" "acr_suffix" {
  byte_length = 4
}

output "acr_name" {
  value = azurerm_container_registry.wine_acr.name
}

# 7. Define our Azure Container Group (The running app)
resource "azurerm_container_group" "wine_app" {
  name                = "wine-predictor-app-instance"
  location            = azurerm_resource_group.wine_rg.location
  resource_group_name = azurerm_resource_group.wine_rg.name
  os_type             = "Linux"
  
  # THIS IS THE CORRECTED LINE:
  ip_address_type     = "Public" 

  # This tells it how to log into our registry to get the image
  image_registry_credential {
    server   = azurerm_container_registry.wine_acr.login_server
    username = azurerm_container_registry.wine_acr.admin_username
    password = azurerm_container_registry.wine_acr.admin_password
  }

  # This defines our actual container
  container {
    name   = "wine-predictor-container"
    image  = "winepredictoracra2ff2145.azurecr.io/wine-predictor:latest"
    cpu    = 1
    memory = 2
    ports {
      port     = 8501
      protocol = "TCP" # Added protocol for clarity
    }
  }
}

# 8. Output the final URL of our running app
output "app_url" {
  # THIS IS THE CORRECTED OUTPUT:
  value = "http://${azurerm_container_group.wine_app.ip_address}:8501"
} 