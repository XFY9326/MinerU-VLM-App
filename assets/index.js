function setPDFWorker() {
  pdfjsLib.GlobalWorkerOptions.workerSrc = "/res/gradio_pdf/pdf.worker.min.mjs";
}
