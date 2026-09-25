# Instalación rápida (PC del laboratorio u otra computadora)

Instala el pipeline y su **interfaz gráfica** en una computadora nueva. El
mismo código funciona con GPU **NVIDIA (CUDA)**, **AMD (ROCm, solo Linux)** o
sin GPU (CPU, unas 40× más lento). El instalador detecta cuál hay y pone la
versión correcta de PyTorch.

## Requisitos

- **Python 3.10 o más nuevo.** En Windows, desde [python.org](https://www.python.org),
  marcando *Add python.exe to PATH*.
- **Git** y acceso a GitHub (el repositorio es de la organización). La
  forma más simple es `gh auth login` con la cuenta que tiene acceso.
- **NVIDIA:** el driver instalado (`nvidia-smi` tiene que responder). No
  hace falta instalar el CUDA Toolkit: PyTorch trae lo que necesita.
  El instalador usa CUDA 13.0 si el driver lo soporta (≥ 580) y, si no,
  CUDA 12.6. Las dos funcionan con la RTX 3070.
- ~6 GB libres (PyTorch con CUDA ~3 GB, modelo Cellpose-SAM ~1.2 GB).

## Pasos

### Linux

```bash
git clone https://github.com/AFMMenchacalab/pipeline-migracion-celular.git
cd pipeline-migracion-celular
git switch experimental/interfaz-cuda      # mientras esto no esté en main
./instalar.sh                              # o --cuda / --rocm / --cpu para forzar
./iniciar_interfaz.sh
```

### Windows

```powershell
git clone https://github.com/AFMMenchacalab/pipeline-migracion-celular.git
cd pipeline-migracion-celular
git switch experimental/interfaz-cuda
```

Después, doble clic en **`instalar.bat`** y, cuando termine, en
**`iniciar_interfaz.bat`**.

El instalador termina corriendo `scripts/verificar_instalacion.py`, que
muestra la GPU detectada, descarga el modelo, segmenta una imagen de prueba
y prueba el tracking. Si dice **TODO BIEN**, está listo.

## Uso de la interfaz

Se abre sola en el navegador (`http://127.0.0.1:8080`).

1. **Recepción desde la Raspberry.** *Iniciar recepción*. La interfaz
   muestra un **código de 6 dígitos**. En la Pi (panel *Envío a
   computadora* de MicroscopeOS, rama `feature/usb-envio-pc`): **Buscar
   computadoras en la red**, elegir esta PC de la lista, escribir el código y
   *Probar conexión*. Se hace una sola vez: el código queda guardado en las
   dos, y si la PC cambia de IP la Pi la vuelve a encontrar sola por su
   nombre. Si la búsqueda no la encuentra (redes distintas o Wi-Fi que aísla
   equipos), la interfaz muestra la dirección para escribirla a mano.
2. **Segmentación en vivo.** Elegir la imagen para Cellpose y *Iniciar
   segmentación*. Cada ciclo que llega se segmenta en unos segundos; la
   última imagen con los contornos se ve en *Segmentación en vivo*.
3. **Experimentos.** Elegir uno de la lista: progreso, células por ciclo y
   el botón **Analizar trayectorias**, que hace el tracking y calcula
   rapidez, direccionalidad, α, entropía (EAD₁) y el MSD con su ajuste.
4. **Carpeta de imágenes.** Para analizar un experimento que no llegó por
   red (por ejemplo, copiado de una memoria USB), elegir la carpeta que
   contiene las carpetas `timelapse_…` y usar el paso 2 y 3 igual.

Opciones: `iniciar_interfaz.sh --publico` la deja visible desde otras
computadoras de la red (por defecto solo desde la misma PC).

**Cortafuegos:** para recibir de la Pi hay que permitir en la red local el
puerto **8765/tcp** (imágenes) y el **8766/udp** (búsqueda automática). En
Windows basta con aceptar el aviso de Windows Defender la primera vez
(marcar *Redes privadas*). En Linux con firewalld:
`sudo firewall-cmd --add-port=8765/tcp --add-port=8766/udp --permanent && sudo firewall-cmd --reload`.

## Dónde queda todo

| Qué | Dónde |
|---|---|
| Imágenes recibidas | `datasets/microscopio_propio/<experimento>/cam<N>/` |
| Máscaras | `~/microscopio_cache/masks/propio/<experimento>/cam<N>/` |
| Resultados del análisis | `~/microscopio_cache/analisis/propio/<experimento>/cam<N>/` (`tracks.csv`, `por_celula.csv`, `resumen.json`, figuras) |

## Sin interfaz (consola)

Los mismos pasos existen como scripts: `31_receptor_microscopio.py`,
`32_segmentar_en_vivo.py`; el análisis se puede llamar desde Python con
`lib.propio.analizar(...)`.
