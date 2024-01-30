//
// Created by evaletov on 1/28/24.
//

#ifndef SERIALIZABLE_BASE_H
#define SERIALIZABLE_BASE_H

#include <serial/eoSerial.h>
#include <string>

namespace glyfada_parallel { // Replace with your actual namespace

    template<class T>
    class SerializableBase : public eoserial::Persistent {
    public:
        virtual ~SerializableBase();

        operator T&();

        // Method to re-initialize the _value member
        void setValue(const T& newValue);

        SerializableBase();
        SerializableBase(T base);

        void unpack(const eoserial::Object* obj) override;
        eoserial::Object* pack() const override;

    private:
        T _value;
    };

} // namespace your_namespace

#include "SerializableBase.tpp"

#endif // SERIALIZABLE_BASE_H