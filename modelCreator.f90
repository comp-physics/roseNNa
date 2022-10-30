module model
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE readTester
    ! ===============================================================

    
    IMPLICIT NONE
    contains
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
        
        SUBROUTINE use_model(        i0, i1, i2,         o0)
            IMPLICIT NONE
            !INPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 2, 5) :: i0
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 1, 2) :: i1
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 1, 2) :: i2
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:,:,:) :: output0
            !================================================

            !OUTPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(OUT), DIMENSION(        1, 1, 2) :: o0
            !===========================

            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: input
            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: hidden_state
            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: cell_state

            REAL :: T1, T2

            ALLOCATE(input(        1, 2, 5))
            ALLOCATE(hidden_state(        1, 1, 2))
            ALLOCATE(cell_state(        1, 1, 2))

            input = i0
            hidden_state = i1
            cell_state = i2

            
            CALL CPU_TIME(T1)
            
            !========Transpose============
            input = RESHAPE(input,(/SIZE(input, dim = 2), SIZE(input, dim = 1), SIZE(input, dim = 3)/), order = [2, 1, 3])

            !========LSTM Layer============
            CALL lstm(input, hidden_state, cell_state, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh,&
                & output0)

            call CPU_TIME(T2)
            o0 = cell_state
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
