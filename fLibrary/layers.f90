module model_layers

    USE activation_functions
    USE derived_types
    use iso_c_binding

    implicit none

contains
    !=== for Gemm operations ======
    subroutine linear_layer(inp, lin, transInp)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: transInp
        REAL (c_double), ALLOCATABLE, intent(inout) :: inp(:,:) !===input is 2d usually (k,n), where n is usually 1
        TYPE(linLayer), INTENT(IN) :: lin !===stores the weights (m,k) and biases (m,1)
        REAL (c_double), DIMENSION(size(lin%weights,1),size(inp,transInp+1)) :: bias_broadcast !==(m,n)
        bias_broadcast = SPREAD(lin%biases, 2, size(inp,transInp+1))
        if (transInp == 0) THEN
            inp = matmul(inp,TRANSPOSE(lin%weights)) + TRANSPOSE(bias_broadcast)
        ELSE
            inp = matmul(lin%weights,inp) + bias_broadcast
        END IF
    end subroutine

    subroutine matmul2D(inp1, inp2)
        IMPLICIT NONE
        REAL (c_double), allocatable, intent(inout), dimension(:,:) :: inp1
        REAL (c_double), intent(in), dimension(:,:) :: inp2

        inp1 = matmul(inp1,inp2)
    end subroutine

    subroutine matmul3D(inp1, inp2)
        IMPLICIT NONE
        REAL (c_double), allocatable, intent(inout), dimension(:,:,:) :: inp1
        REAL (c_double), intent(in), dimension(:,:,:) :: inp2
        REAL (c_double), dimension(size(inp1,1),size(inp1,2),size(inp2,3)) :: out
        integer :: i

        DO i=1, size(inp1,1)
            out(i,:,:) = MATMUL(inp1(i,:,:),inp2(i,:,:))
        END DO
        inp1 = out
    end subroutine

    subroutine matmul4D(inp1, inp2)
        IMPLICIT NONE
        REAL (c_double), allocatable, intent(inout), dimension(:,:,:,:) :: inp1
        REAL (c_double), intent(in), dimension(:,:,:,:) :: inp2
        REAL (c_double), dimension(size(inp1,1),size(inp1,2),size(inp1,3),size(inp2,4)) :: out
        integer :: i
        integer :: j

        DO i=1, size(inp1,1)
            DO j=1,size(inp1,2)
                out(i,j,:,:) = MATMUL(inp1(i,j,:,:),inp2(i,j,:,:))
            END DO
        END DO
        inp1 = out
    end subroutine

    subroutine lstm_cell(input, hid1, cell1, Whh, Wih, Bih, Bhh)
        implicit none
        REAL (c_double), intent(in), DIMENSION(:,:) :: input !== (n,batch_size)
        REAL (c_double), intent(inout), ALLOCATABLE, DIMENSION(:,:) :: hid1 !==(m,batch_size)
        REAL (c_double), intent(inout), ALLOCATABLE, DIMENSION(:,:) :: cell1 !==(m,batch_size)
        REAL (c_double), intent(in), DIMENSION(:,:) :: Whh !==(4m,m)
        REAL (c_double), intent(in), DIMENSION(:,:) :: Wih !==(4m,n)
        REAL (c_double), intent(in), DIMENSION(:) :: Bhh !==(4m,1)
        REAL (c_double), intent(in), DIMENSION(:) :: Bih !==(4m,1)
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: hiddenOut !==(m,batch_size)
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: cellOut


        REAL (c_double), DIMENSION(size(Bhh, 1), size(input,2)) :: Bhh_broadcast !== (4m, batch_size)
        REAL (c_double), DIMENSION(size(Bhh, 1), size(input,2)) :: Bih_broadcast
        REAL (c_double), DIMENSION(size(Whh, dim=1), size(input,2)) :: gates_out
        REAL (c_double), DIMENSION(size(hid1,1),4,size(input,2)) :: chunks !4, size(Whh, dim=1)/4
        ALLOCATE(hiddenOut(size(hid1),size(input,2)))
        ALLOCATE(cellOut(size(cell1),size(input,2)))

        !======= applying bhh and bih to each column of input =====
        Bhh_broadcast = SPREAD(Bhh, 2, size(input,2))
        Bih_broadcast = SPREAD(Bih, 2, size(input,2))

        gates_out = MATMUL(Wih, input) + Bih_broadcast + MATMUL(Whh, hid1) + Bhh_broadcast !==(4m,batch_size) BROADCAST BIAS

        chunks = RESHAPE(gates_out, (/size(gates_out,1)/4, 4, size(input,2)/))

        chunks(:, 1, :) = sigmoid2d(chunks(:, 1, :))
        chunks(:, 2, :) = sigmoid2d(chunks(:, 2, :))
        chunks(:, 3, :) = tanhh2d(chunks(:, 3, :))
        chunks(:, 4, :) = sigmoid2d(chunks(:, 4, :))

        cellOut = (chunks(:, 2, :) * cell1) + (chunks(:, 1, :) * chunks(:, 3, :))
        hiddenOut = chunks(:, 4, :) * tanhh2d(cellOut)
        hid1 = hiddenOut
        cell1 = cellOut
    end subroutine

    subroutine lstm(input, hid1, cell1, Whh, Wih, Bih, Bhh, output)
        implicit none
        REAL (c_double), intent(inout), ALLOCATABLE, DIMENSION(:,:,:) :: input !== (timesteps,batch_size,n)
        REAL (c_double), intent(inout), ALLOCATABLE, DIMENSION(:,:,:) :: hid1 !==(num_directions,batch_size,m), add another dim for num dir
        REAL (c_double), intent(inout), ALLOCATABLE, DIMENSION(:,:,:) :: cell1 !==(num_directions,batch_size,m), add another dim for num dir
        REAL (c_double), intent(in), ALLOCATABLE, DIMENSION(:,:,:) :: Whh !==(num_directions,4m,m), add another dim for num dir
        REAL (c_double), intent(in), ALLOCATABLE, DIMENSION(:,:,:) :: Wih !==(num_directions,4m,n), add another dim for num dir
        REAL (c_double), intent(in), ALLOCATABLE, DIMENSION(:) :: Bhh !==(4m,1)
        REAL (c_double), intent(in), ALLOCATABLE, DIMENSION(:) :: Bih !==(4m,1)
        REAL (c_double), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: output !==(timesteps,num_directions,m,batch_size)
        ! INTEGER, INTENT(IN) :: nlayers, NEED TO ADD NLAYERS FUNCTIONALITY (no need for now since onnx apparently doesn't support it)
        INTEGER :: timesteps
        INTEGER :: i
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: hid1changed
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: cell1changed
        timesteps = size(input,1)
        input = reshape(input, (/SIZE(input,dim=1),SIZE(input,dim=3), SIZE(input,dim=2)/), order = [1,3,2])
        hid1 = reshape(hid1, (/SIZE(hid1,dim=1),SIZE(hid1,dim=3), SIZE(hid1,dim=2)/), order = [1,3,2])
        cell1 = reshape(cell1, (/SIZE(cell1,dim=1),SIZE(cell1,dim=3), SIZE(cell1,dim=2)/), order = [1,3,2])
        ALLOCATE(output(timesteps,size(hid1,dim=1),size(hid1,dim=2),size(hid1,dim=3)))
        hid1changed = hid1(1,:,:)
        cell1changed = cell1(1,:,:)
        DO i=1, timesteps
            CALL lstm_cell(input(i,:,:), hid1changed, cell1changed, Whh(1,:,:), Wih(1,:,:), Bih, Bhh)
            hid1(1,:,:) = hid1changed
            cell1(1,:,:) = cell1changed
            output(i,:,:,:) = hid1
        END DO
        hid1 = reshape(hid1, (/SIZE(hid1,dim=1),SIZE(hid1,dim=3), SIZE(hid1,dim=2)/), order = [1,3,2]) !==reshaped to (num_directions,batch_size,m)
        cell1 = reshape(cell1, (/SIZE(cell1,dim=1),SIZE(cell1,dim=3), SIZE(cell1,dim=2)/), order = [1,3,2]) !==(num_directions,batch_size,m)
        output = reshape(output,(/SIZE(output,dim=1),SIZE(output,dim=2), SIZE(output,dim=4),SIZE(output,dim=3)/), order = [1,2,4,3]) !==(timesteps,num_directions,batch_size,m)
    end subroutine

    function padding(arr, input)
        implicit none
        integer, dimension(:), intent(in) :: arr
        REAL (c_double), dimension(:,:,:,:), intent(in) :: input
        REAL (c_double), dimension(size(input,1),size(input,2),size(input,3)+2*arr(1),size(input,4)+2*arr(2)) :: padding
        REAL (c_double), dimension(size(input,1),size(input,2),size(input,3)+2*arr(1),size(input,4)+2*arr(2)) :: formatted
        
        integer :: i
        integer :: j
        integer :: k
        integer :: numIts 
        integer :: channels
        integer :: rows
        integer :: cols
        integer :: inputrows
        DO numIts=1, size(input,1)
            DO channels=1, size(formatted,2) !iterate over channels
                DO i=1, arr(1)
                    DO j=1, size(formatted,4)
                        formatted(numIts,channels,i,j) = 0
                    END DO
                END DO
                inputrows=1
                DO rows = 1+arr(1), 1+arr(1)+size(input,3)-1
                    DO k=1, arr(2)
                        formatted(numIts,channels,rows,k) = 0
                    END DO
        
                    DO cols=1, size(input,4)
                        formatted(numIts,channels,rows,cols+arr(2)) = input(numIts,channels,inputrows,cols)
                    END DO
        
        
                    DO k=size(formatted,4) - arr(2)+1, size(formatted,4)
                        formatted(numIts,channels,rows,k) = 0
                    END DO
                    inputrows = inputrows + 1
                END DO
        
                DO i=size(formatted,3)-arr(1)+1, size(formatted,3)
                    DO j=1, size(formatted,4)
                        formatted(numIts,channels,i,j) = 0
                    END DO
                END DO
            END DO
        END DO
        padding = formatted
    end function padding

    subroutine conv(inp, convWeights, bias, dilations, pads, strides)
        implicit none
        REAL (c_double), INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: inp !==(batches,numImages,imageD1,imageD2)
        REAL (c_double), INTENT(IN), ALLOCATABLE, DIMENSION(:,:,:,:) :: convWeights !==(numConvRows,numConvCols,ConvRowDim,ConvColDim)
        REAL (c_double), INTENT(IN), ALLOCATABLE, DIMENSION(:) :: bias
        INTEGER, INTENT(IN), DIMENSION(:) :: dilations
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        INTEGER :: in_channels !==numImages SHOULD BE INTENT(IN)
        INTEGER :: out_channels !==numConvCols SHOULD BE INTENT(IN)
        INTEGER :: kernel_size !==(ConvRowDim,ConvColDim) SHOULD BE INTENT(IN)
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: out

        REAL (c_double), DIMENSION(size(inp,1),size(inp,2),size(inp,3)+2*pads(1),size(inp,4)+2*pads(2)) :: padded

        INTEGER :: outer
        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        INTEGER :: numBatches
        INTEGER :: itBatches
        REAL (c_double) :: sumini = 0
        numBatches = size(inp,1)
        in_channels = SIZE(inp, dim=2)
        out_channels = SIZE(convWeights, dim=1)
        kernel_size = SIZE(convWeights, dim=3)
        padded = padding(pads, inp)
        ALLOCATE(out(numBatches,out_channels, (size(padded,dim=3)-kernel_size)/strides(1) + 1, &
            (size(padded,dim=4)-kernel_size)/strides(2)+1))
        outRowDim = size(out,4)
        outColDim = size(out,3)

        DO itBatches = 1, numBatches
            DO outer = 0, out_channels-1 !==iterating through each output image
                DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
                    DO inner = 0, in_channels-1 !==applying kernel to each input image
                        sumini = sumini + SUM(padded(itBatches,inner+1, &
                        (1 + (overImage/outRowDim)*strides(1)):((overImage/outRowDim)*strides(1)+kernel_size) &
                        ,(1 + MODULO(overImage,outRowDim)*strides(2)):(MODULO(overImage,outRowDim)*strides(2)+kernel_size)) &
                            * convWeights(outer+1,inner+1,:,:)) !==depending on the way convWeights is laid out change position of outer/inner, currently it is row based (input images are applied to 1st row then 2nd row, etc.)
                    END DO
                    out(itBatches,outer+1,overImage/outRowDim + 1,MODULO(overImage,outRowDim)+1) = sumini + bias(outer+1)
                    sumini = 0
                END DO
            END DO
        END DO
        inp = out
        DEALLOCATE(out)

    END subroutine


    subroutine max_pool(inp, maxpool, ceil_mode, pads, strides) !==ceil_mode, pads, strides
        implicit none
        REAL (c_double), INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: inp !==(numImages,imageD1,imageD2)
        TYPE(maxpoolLayer), INTENT(IN) :: maxpool !==(ConvRowDim,ConvColDim)
        INTEGER, INTENT(IN) :: ceil_mode
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        !==INTEGER, INTENT(IN) :: stride IMPLEMENT THIS
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: out
        INTEGER :: kernel_size

        REAL (c_double), DIMENSION(size(inp,1),size(inp,2),size(inp,3)+2*pads(1),size(inp,4)+2*pads(2)) :: padded

        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        padded = padding(pads, inp)
        kernel_size = maxpool%kernel_size
        ALLOCATE(out(1,size(padded,dim=2), (size(padded,dim=3)-kernel_size)/strides(1) + 1, &
            (size(padded,dim=4)-kernel_size)/strides(2)+1))
        outRowDim = size(out,4)
        outColDim = size(out,3)

        DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
            DO inner = 0, size(padded,dim=2)-1 !==applying kernel to each input image
                out(1,inner+1,(overImage/outRowDim)+1,MODULO(overImage,outColDim)+1) = &
                 MAXVAL(padded(1,inner+1,1 + ((overImage/outRowDim)*strides(1)):((overImage/outRowDim)*strides(1)+kernel_size) &
                 ,(1 + MODULO(overImage,outRowDim)*strides(2)): &
                 (MODULO(overImage,outRowDim)*strides(2)+kernel_size)))
            END DO
        END DO
        inp = out
        DEALLOCATE(out)

    end subroutine

    subroutine avgpool(inp, maxpool, ceil_mode, pads, strides) !==ceil_mode, pads, strides
        implicit none
        REAL (c_double), INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: inp !==(numImages,imageD1,imageD2)
        TYPE(avgpoolLayer), INTENT(IN) :: maxpool !==(ConvRowDim,ConvColDim)
        INTEGER, INTENT(IN) :: ceil_mode
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        !==INTEGER, INTENT(IN) :: stride IMPLEMENT THIS
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: out
        INTEGER :: kernel_size
        INTEGER :: total

        REAL (c_double), DIMENSION(size(inp,1),size(inp,2),size(inp,3)+2*pads(1),size(inp,4)+2*pads(2)) :: padded

        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        padded = padding(pads, inp)
        kernel_size = maxpool%kernel_size
        total = kernel_size * kernel_size
        ALLOCATE(out(1,size(padded,dim=2), (size(padded,dim=3)-kernel_size)/strides(1) + 1, &
            (size(padded,dim=4)-kernel_size)/strides(2)+1))
        outRowDim = size(out,4)
        outColDim = size(out,3)

        DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
            DO inner = 0, size(padded,dim=2)-1 !==applying kernel to each input image
                out(1,inner+1,(overImage/outRowDim)+1,MODULO(overImage,outColDim)+1) = &
                 SUM(padded(1,inner+1,1 + ((overImage/outRowDim)*strides(1)):((overImage/outRowDim)*strides(1)+kernel_size) &
                 ,(1 + MODULO(overImage,outRowDim)*strides(2)): &
                 (MODULO(overImage,outRowDim)*strides(2)+kernel_size)))/total
            END DO
        END DO
        inp = out
        DEALLOCATE(out)

    end subroutine

    !
    ! != INSTEAD WHAT I SHOULD DO IS DO ALL OF THIS IN MODELCREATOR.FPP
    ! !== BASICALLY, I TAKE THE SMALLER ARRAY, WHICH SHOULD BE THE ONE THATS ADDLAYER%ADDER
    ! !== I SPREAD THIS BASED ON EACH DIMENSION OF THE LARGER ARRAY
    ! !==THEN I RESHAPE TO THE SHAPE OF THE LARGER ARRAY
    ! subroutine ad(inp, adding, reshapeDim)
    !     INTEGER, INTENT(in) :: reshapeDim
    !     REAL (c_double), ALLOCATABLE, intent(in) :: inp(:,:,:,:)
    !     TYPE(addLayer), INTENT(IN) :: adding
    !     REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: intermediate
    !     INTEGER, DIMENSION(4) :: true
    !     INTEGER, DIMENSION(reshapeDim) :: reshaped
    !     INTEGER, ALLOCATABLE, DIMENSION(:) :: inter
    !     INTEGER, DIMENSION(4) :: changing
    !     INTEGER :: i
    !     true = SHAPE(inp)
    !     intermediate = adding%adder
    !     inter = SHAPE(intermediate)
    !     DO i=1, 4
    !         if (i .ne. 4) then
    !             changing = [true(:i),inter((i+1):)]
    !             intermediate = RESHAPE(SPREAD(intermediate,i,size(inp,i)), changing)
    !         else
    !             intermediate = RESHAPE(SPREAD(intermediate,i,size(inp,i)), SHAPE(inp))
    !         end if
    !     END DO

    !     if (reshapeDim .eq. 1) then
    !         asdf
    !     else if (reshapeDim .eq. 2) then
    !         asdf
    !     else if (reshapeDim .eq. 3) then
    !         asdf
    !     else
    !         asdf
    !     end if 
    !     print *, RESHAPE(intermediate, )  ! make var called output
    ! end subroutine
    !hi
    function broadc(inp, trueShape, spreadInfo) result(out)
        implicit none
        REAL (c_double), dimension(:,:,:,:), intent(in) :: inp
        integer, dimension(4) :: trueShape
        integer, dimension(:,:), intent(in) :: spreadInfo
        REAL (c_double), ALLOCATABLE, dimension(:,:,:,:) :: out
    
        INTEGER, DIMENSION(4) :: true
        INTEGER, ALLOCATABLE, DIMENSION(:) :: inter
        INTEGER, DIMENSION(4) :: changing
        REAL (c_double), ALLOCATABLE, dimension(:,:,:,:) :: intermediate
    
        INTEGER :: i
        intermediate = inp
        inter = SHAPE(inp)
        DO i=1, size(spreadInfo,1)
            if (i .ne. size(spreadInfo,1)) then
                changing = [trueShape(:spreadInfo(i,1)),inter((spreadInfo(i,1)+1):)]
                intermediate = RESHAPE(SPREAD(intermediate,spreadInfo(i,1),spreadInfo(i,2)), changing)
            else
                intermediate = RESHAPE(SPREAD(intermediate,spreadInfo(i,1),spreadInfo(i,2)), trueShape)
            end if
        END DO
        out = intermediate
    end function


end module model_layers
