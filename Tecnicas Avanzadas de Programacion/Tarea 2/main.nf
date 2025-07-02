#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Proceso que emite un saludo por stdout
process process_saludo {
    output:
    stdout

    script:
    """
    echo "Hola Mundo desde Nextflow!"
    """
}
//PRoceso esperar tiempo
process esperar_tiempo {
    tag "espera_60s"

    output:
    stdout

    script:
    """
    echo "Iniciando espera de 60 segundos..."
    sleep 60
    echo "Proceso completado. 🎉"
    """
}

// Definir workflow y capturar salida
workflow {
    saludo_ch = process_saludo()
    saludo_ch.view()
    resultado = esperar_tiempo()
    resultado.view()
}

