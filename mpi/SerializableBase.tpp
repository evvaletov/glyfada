//
// Created by evaletov on 1/28/24.
//

// SerializableBase.tpp

namespace glyfada_parallel {

    template<class T>
    SerializableBase<T>::SerializableBase() : _value() {
        // Implementation...
    }

    template<class T>
    SerializableBase<T>::SerializableBase(T base) : _value(base) {
        // Implementation...
    }

    template<class T>
    SerializableBase<T>::~SerializableBase() = default;

    template<class T>
    SerializableBase<T>::operator T&() {
        return _value;
    }

    template<class T>
    void SerializableBase<T>::setValue(const T& newValue) {
        _value = newValue;
    }

    template<class T>
    void SerializableBase<T>::unpack(const eoserial::Object* obj) {
        eoserial::unpack(*obj, "value", _value);
    }

    template<class T>
    eoserial::Object* SerializableBase<T>::pack() const {
        eoserial::Object* obj = new eoserial::Object;
        obj->add("value", eoserial::make(_value));
        return obj;
    }

} // namespace glyfada_parallel
// namespace your_namespace
