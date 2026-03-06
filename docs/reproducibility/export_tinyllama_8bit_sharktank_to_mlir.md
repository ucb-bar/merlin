# Exporting TinyLlama-1.1B (8-Bit Weights) to MLIR using SHARK Tank

This guide documents the step-by-step process to convert the TinyLlama-1.1B model into an 8-bit quantized GGUF file and then export it to the MLIR format using the SHARK Tank toolkit, following the GGUF LLM workflow. It includes troubleshooting steps for common dependency issues encountered during the process.

**Target:** Produce a `.mlir` file representing TinyLlama-1.1B with 8-bit weights and float activations, suitable for compilation with IREE.

**Model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

**Frameworks/Tools:**
* SHARK Tank ([nod-ai/shark-ai GitHub - Assumed](https://github.com/nod-ai/shark-ai/tree/main/sharktank) - *Note: Official link might differ*)
* llama.cpp ([ggerganov/llama.cpp GitHub](https://github.com/ggerganov/llama.cpp))
* Hugging Face Hub CLI ([Installation Guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli))
* IREE Compiler ([iree-org/iree GitHub](https://github.com/iree-org/iree))

## Prerequisites

Ensure you have the following installed:
* Python (>= 3.9 recommended)
* `pip` and `venv`
* `git`
* `cmake`
* A C/C++ compiler toolchain (`build-essential` on Debian/Ubuntu, Xcode Command Line Tools on macOS, etc.)

## Step 1: Set Up SHARK Tank Environment

Set up a Python virtual environment and install SHARK Tank and its dependencies.

```bash
# 1. Create a Python virtual environment (e.g., in your home directory)
# Replace <path_to_shark_ai_repo> with the actual path to your cloned repo
# If you haven't cloned it yet, do so first:
# git clone [https://github.com/nod-ai/shark-ai.git](https://github.com/nod-ai/shark-ai.git) <path_to_shark_ai_repo>

python -m venv --prompt sharktank ~/sharktank-venv

# 2. Activate the environment
source ~/sharktank-venv/bin/activate

# 3. Navigate to your SHARK Tank repository
# cd <path_to_shark_ai_repo>

# 4. Install base and SHARK Tank dependencies
# (Ensure requirements.txt files exist at these paths)
pip install -r requirements.txt
pip install -r sharktank/requirements.txt
pip install -e sharktank/
pip install -e shortfin/ # If shortfin directory exists

# 5. Install the 'wave_lang' dependency (needed for kernels)
# This addresses a potential ModuleNotFoundError
pip install wave_lang

# 6. Install compatible PyTorch, torchvision, and Triton
# This addresses potential RuntimeError/ImportError issues
# Note: Use the stable versions first. If issues persist, consider nightlies carefully.
pip install --upgrade torch torchvision triton

# 7. Log in to Hugging Face CLI (needed for downloading models)
huggingface-cli login
```

## Step 2: Set Up llama.cpp Toolchain

Clone and build llama.cpp to get the necessary conversion and quantization tools.

```bash
# 1. Clone the llama.cpp repository (e.g., in your home directory)
git clone [https://github.com/ggerganov/llama.cpp.git](https://github.com/ggerganov/llama.cpp.git) ~/llama.cpp

# 2. Navigate into the directory
cd ~/llama.cpp

# 3. Install Python dependencies for the conversion script
# (Ensure your sharktank-venv is active)
# source ~/sharktank-venv/bin/activate
pip install -r requirements.txt

# 4. Create a build directory and navigate into it
mkdir build
cd build

# 5. Configure the build using cmake
cmake ..

# 6. Compile the project (this builds the tools like 'llama-quantize')
cmake --build .

# 7. Navigate back to your original project directory (e.g., where shark-ai repo is)
# cd <path_to_shark_ai_repo> # Or your preferred working directory
```

## Step 3: Download, Convert, Quantize, and Export TinyLlama

Execute the model processing workflow. Ensure your sharktank-venv is active.

**A. Download TinyLlama Base Model**

```bash
# Download the TinyLlama model files to a temporary directory
hf download --local-dir /tmp/TinyLlama-1.1B \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**B. Convert to FP16 GGUF**

```bash
python ~/llama.cpp/convert_hf_to_gguf.py --outtype f16 \
  --outfile /tmp/TinyLlama-1.1B-f16.gguf \
  /tmp/TinyLlama-1.1B
```

**C. Quantize to 8-Bit (Q8_0) GGUF**

```bash
~/llama.cpp/build/bin/llama-quantize --pure \
  /tmp/TinyLlama-1.1B-f16.gguf \
  /tmp/TinyLlama-1.1B-q8_0.gguf Q8_0
```

**D. Export 8-Bit GGUF to MLIR**

```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file=/tmp/TinyLlama-1.1B-q8_0.gguf \
  --output-mlir=/tmp/TinyLlama-1.1B-q8_0.mlir \
  --output-config=/tmp/TinyLlama-1.1B-q8_0.json
```

**Step 4: Compile MLIR to VMFB**

Compile the generated .mlir file into an IREE deployable format (.vmfb).

```bash
iree-compile /tmp/TinyLlama-1.1B-q8_0.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu-features=host \
  -o /tmp/TinyLlama-1.1B-q8_0_cpu.vmfb
```