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
        SUBROUTINE use_model(Input3, Parameter5, Parameter6, Parameter87, Parameter88, Pooling160_Output_0_reshape0_shape,&
            & Parameter193, Parameter193_reshape1_shape, Parameter194, Plus214_Output_0)
            IMPLICIT NONE
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: Input3
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: Parameter5
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:) :: Parameter6
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: Parameter87
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:) :: Parameter88
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:) :: Pooling160_Output_0_reshape0_shape
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: Parameter193
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:) :: Parameter193_reshape1_shape
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:) :: Parameter194
            REAL, ALLOCATABLE,  DIMENSION(:,:) :: output0
            REAL, ALLOCATABLE,  DIMENSION(:,:) :: output1
            REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:) :: Plus214_Output_0
            REAL :: T1, T2
            CALL CPU_TIME(T1)
            
            !========Reshape============
            Parameter193 = RESHAPE(Parameter193,(/SIZE(Parameter193, dim = 4), SIZE(Parameter193, dim = 3), SIZE(Parameter193, dim&
                & = 2), SIZE(Parameter193, dim = 1)/), order = [4, 3, 2, 1])
            output0 = RESHAPE(Parameter193,(/256, 10/), order = [2, 1])

            !========Conv Layer============
            CALL conv(Input3, convLayers(1)%weights, convLayers(1)%biases,         (/5, 5/),         (//),         (/1, 1/))

            !===========Add============
            Input3 = Input3 + RESHAPE(broadc(addLayers(1)%adder,        (/1, 8, 28, 28/),RESHAPE(        (/3, 1, 4, 1/),       &
                & (/2, 2/), order=[2,1])),         (/1, 8, 28, 28/))

            !========MaxPool Layer============
            CALL max_pool(Input3,maxpoolLayers(1), 0,         (/0, 0, 0, 0/),         (//))

            !========Conv Layer============
            CALL conv(Input3, convLayers(2)%weights, convLayers(2)%biases,         (/5, 5/),         (//),         (/1, 1/))

            !===========Add============
            Input3 = Input3 + RESHAPE(broadc(addLayers(2)%adder,        (/1, 16, 14, 14/),RESHAPE(        (/3, 1, 4, 1/),       &
                & (/2, 2/), order=[2,1])),         (/1, 16, 14, 14/))

            !========MaxPool Layer============
            CALL max_pool(Input3,maxpoolLayers(2), 0,         (/0, 0, 0, 0/),         (//))

            !========Reshape============
            Input3 = RESHAPE(Input3,(/SIZE(Input3, dim = 4), SIZE(Input3, dim = 3), SIZE(Input3, dim = 2), SIZE(Input3, dim =&
                & 1)/), order = [4, 3, 2, 1])
            output1 = RESHAPE(Input3,(/1, 256/), order = [2, 1])

            !===========Add============
            output1 = output1 + RESHAPE(broadc(addLayers(3)%adder,        (/1, 1, 1, 10/),RESHAPE(        (//),        (/0, 2/),&
                & order=[2,1])),         (/1, 10/))

            call CPU_TIME(T2)
            print *, "-------------"
            Plus214_Output_0 = output1
            print *, "TIME TAKEN:", T2-T1
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
