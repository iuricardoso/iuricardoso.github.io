classDiagram
    class Sensor {
        <<interface>>
        +iniciar() void
        +ler() float
        +getMinimo() float
        +getMaximo() float
        +~Sensor()
    }

    class Display {
        <<interface>>
        +iniciar() void
        +exibir(valorMapeado: int) void
        +getResolucao() int
        +~Display()
    }

    class SensorTemperaturaNTC {
        -pino: uint8_t
        -BETA: float
        +SensorTemperaturaNTC(pinoAnalogico: uint8_t)
        +iniciar() void
        +ler() float
        +getMinimo() float
        +getMaximo() float
    }

    class SensorUltrassonicoHCSR04 {
        -pinoTrig: uint8_t
        -pinoEcho: uint8_t
        +SensorUltrassonicoHCSR04(trig: uint8_t, echo: uint8_t)
        +iniciar() void
        +ler() float
        +getMinimo() float
        +getMaximo() float
    }

    class DisplayBarraLeds {
        -pinos: uint8_t*
        -qtdLeds: int
        +DisplayBarraLeds(pinosLeds: uint8_t*, quantidade: int)
        +iniciar() void
        +exibir(valor: int) void
        +getResolucao() int
    }

    class DisplayLCD {
        -lcd: LiquidCrystal_I2C
        -colunas: int
        +DisplayLCD(enderecoI2C: uint8_t, cols: int, lins: int)
        +iniciar() void
        +exibir(valor: int) void
        +getResolucao() int
    }

    class ConversorSensorDisplay {
        -sensor: Sensor*
        -display: Display*
        +ConversorSensorDisplay(s: Sensor*, d: Display*)
        +iniciar() void
        +atualizar() void
    }

    Sensor <|-- SensorTemperaturaNTC : Implementa
    Sensor <|-- SensorUltrassonicoHCSR04 : Implementa
    Display <|-- DisplayBarraLeds : Implementa
    Display <|-- DisplayLCD : Implementa
    ConversorSensorDisplay o-- Sensor : Agrega
    ConversorSensorDisplay o-- Display : Agrega