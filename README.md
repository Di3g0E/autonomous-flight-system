# autonomous-flight-system

## Primeros Pasos

Sigue estas instrucciones para configurar el entorno y ejecutar el simulador.

1.  **Navega al directorio del simulador**
    Desde la raíz del proyecto, entra en el directorio `UAV_3D_Virtual_Env-master`:
    ```bash
    cd UAV_3D_Virtual_Env-master
    ```

2.  **Crea el entorno de Conda**
    Usa el archivo `environment.yml` para crear el entorno con todas las dependencias necesarias:
    ```bash
    conda env create --name <nombre_del_entorno> -f environment.yml
    ```

3.  **Activa el entorno**
    Activa el entorno que acabas de crear. El nombre del entorno (busca la línea `name:`) está especificado dentro del archivo `environment.yml`.
    ```bash
    conda activate <nombre_del_entorno>
    ```

4.  **Ejecuta el simulador**
    Una vez activado el entorno y asegurándote de estar en el directorio `UAV_3D_Virtual_Env-master` (debido a las referencias relativas internas), ejecuta el script principal desde tu terminal:
    ```bash
    python main.py
    ```
